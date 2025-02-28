/*
 * Copyright (c) 2011-2012, fortiss GmbH.
 * Licensed under the Apache License, Version 2.0.
 *
 * Use, modification and distribution are subject to the terms specified
 * in the accompanying license file LICENSE.txt located at the root directory
 * of this software distribution. A copy is available at
 * http://chromosome.fortiss.org/.
 *
 * This file is part of CHROMOSOME.
 *
 * $Id$
 */

/**
 * \file
 *         Scheduler abstraction (architecture specific part: Windows).
 *
 * \author
 *         Michael Geisinger <geisinger@fortiss.org>
 */

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "xme/hal/sched.h"

#include "xme/core/component.h"
#include "xme/core/defines.h"
#include "xme/core/resourceManager.h"

#include "xme/hal/random.h"
#include "xme/hal/table.h"
#include "xme/hal/time.h"
#include "xme/hal/tls.h"

#include <stdint.h>
#include <windows.h>

/******************************************************************************/
/***   Type definitions                                                     ***/
/******************************************************************************/
typedef struct
{
	CRITICAL_SECTION threadMutex; ///< Mutex for shared access to the thread that represents this task.
	HANDLE stateChangedEvent; ///< Signal sent to the thread to indicate a task state change.
	HANDLE taskPeriodTimer; ///< Waitable timer for task period.
	HANDLE threadHandle; ///< Handle of the thread that represents the task.
	DWORD threadId; ///< Identifier of the thread that represents the task.
	xme_hal_sched_taskState_t taskState; ///< Execution state of the task (running, suspended, terminating).
	xme_hal_time_interval_t startMs; ///< Start time of the task in milliseconds. If zero, the task is started immediately.
	xme_hal_time_interval_t periodMs; ///< Period of task execution in milliseconds. If zero, the task is only executed once.
	uint8_t priority; ///< Priority of the task. Higher values denote a higher priority.
	xme_hal_sched_taskCallback_t callback; ///< Task callback function.
	xme_core_component_t componentContext; ///< Context of the component this task belongs to.
	void* userData; ///< User data to pass to task callback function.
	bool selfCleanup; ///< Whether the thread itself should clean up its task record upon destruction. This is the case if the thread of the task itself called the removeTask() function, in which case the runtime system must not clean up the resources until the thread has actually terminated.
}
xme_hal_sched_taskDescriptor_t;

typedef struct
{
	//private
	CRITICAL_SECTION taskDescriptorsMutex; ///< Mutex for shared access to the task list.
	XME_HAL_TABLE(xme_hal_sched_taskDescriptor_t, taskDescriptors, XME_HAL_DEFINES_MAX_TASK_DESCRIPTORS); // taskHandle is an index into this table
}
xme_hal_sched_configStruct_t;

/******************************************************************************/
/***   Static variables                                                     ***/
/******************************************************************************/
static xme_hal_sched_configStruct_t xme_hal_sched_config;

/******************************************************************************/
/***   Protoypes                                                            ***/
/******************************************************************************/
DWORD
xme_hal_sched_taskProc(LPVOID lpParameter);

/******************************************************************************/
/***   Implementation                                                       ***/
/******************************************************************************/

xme_core_status_t
xme_hal_sched_init()
{
	XME_HAL_TABLE_INIT(xme_hal_sched_config.taskDescriptors);

	XME_CHECK
	(
		0 != InitializeCriticalSectionAndSpinCount(&xme_hal_sched_config.taskDescriptorsMutex, 0x00000400),
		XME_CORE_STATUS_OUT_OF_RESOURCES
	);

	return XME_CORE_STATUS_SUCCESS;
}

xme_hal_sched_taskHandle_t
xme_hal_sched_addTask
(
	xme_hal_time_interval_t startMs,
	xme_hal_time_interval_t periodMs,
	uint8_t priority,
	xme_hal_sched_taskCallback_t callback,
	void* userData
)
{
	xme_hal_sched_taskHandle_t newTaskHandle;
	xme_hal_sched_taskDescriptor_t* taskDesc;

	XME_CHECK
	(
		NULL != callback,
		XME_HAL_SCHED_INVALID_TASK_HANDLE
	);

	// Synchronize access to the task descriptors mutex
	EnterCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
	{
		// Allocate a unique task handle
		newTaskHandle = (xme_hal_sched_taskHandle_t)XME_HAL_TABLE_ADD_ITEM(xme_hal_sched_config.taskDescriptors);

		XME_CHECK_REC
		(
			XME_HAL_SCHED_INVALID_TASK_HANDLE != newTaskHandle,
			XME_HAL_SCHED_INVALID_TASK_HANDLE,
			{
				LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
			}
		);

		// Store the task profile
		taskDesc = (xme_hal_sched_taskDescriptor_t*)XME_HAL_TABLE_ITEM_FROM_HANDLE(xme_hal_sched_config.taskDescriptors, newTaskHandle);

		XME_ASSERT_RVAL(NULL != taskDesc, XME_HAL_SCHED_INVALID_TASK_HANDLE);

		XME_CHECK_REC
		(
			0 != InitializeCriticalSectionAndSpinCount(&taskDesc->threadMutex, 0x00000400),
			XME_HAL_SCHED_INVALID_TASK_HANDLE,
			{
				XME_HAL_TABLE_REMOVE_ITEM(xme_hal_sched_config.taskDescriptors, (xme_hal_table_rowHandle_t)newTaskHandle);
				LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
			}
		);

		XME_CHECK_REC
		(
			NULL != (taskDesc->stateChangedEvent = CreateEvent(NULL, true, false, NULL)),
			XME_HAL_SCHED_INVALID_TASK_HANDLE,
			{
				CloseHandle(taskDesc->stateChangedEvent);
				DeleteCriticalSection(&taskDesc->threadMutex);
				XME_HAL_TABLE_REMOVE_ITEM(xme_hal_sched_config.taskDescriptors, (xme_hal_table_rowHandle_t)newTaskHandle);
				LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
			}
		);

		// If the task is to be executed periodically or once only with delayed start,
		// initialize a waitable timer that will invoke the task at the appropriate
		// points in time
		if (0 != periodMs || (0 == periodMs && 0 != startMs && XME_HAL_SCHED_TASK_INITIALLY_SUSPENDED != startMs))
		{
			XME_CHECK_REC
			(
				NULL != (taskDesc->taskPeriodTimer = CreateWaitableTimer(NULL, false, NULL)),
				XME_HAL_SCHED_INVALID_TASK_HANDLE,
				{
					CloseHandle(taskDesc->stateChangedEvent);
					DeleteCriticalSection(&taskDesc->threadMutex);
					XME_HAL_TABLE_REMOVE_ITEM(xme_hal_sched_config.taskDescriptors, (xme_hal_table_rowHandle_t)newTaskHandle);
					LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
				}
			);
		}
		else
		{
			taskDesc->taskPeriodTimer = NULL;
		}

		taskDesc->threadHandle = NULL;
		taskDesc->threadId = 0;
		taskDesc->taskState = (XME_HAL_SCHED_TASK_INITIALLY_SUSPENDED == startMs) ? XME_HAL_SCHED_TASK_STATE_SUSPENDED : XME_HAL_SCHED_TASK_STATE_RUNNING;
		taskDesc->startMs = (XME_HAL_SCHED_TASK_INITIALLY_SUSPENDED == startMs) ? 0 : startMs;
		taskDesc->periodMs = periodMs;
		taskDesc->priority = priority;
		taskDesc->callback = callback;
		taskDesc->componentContext = xme_core_resourceManager_getCurrentComponentId();
		taskDesc->userData = userData;
		taskDesc->selfCleanup = false;

		// The last statement of this function related to the thread descriptor is
		// the assignment of the threadHandle member. However, the thread might
		// already access that member before the assignment is complete.
		// Hence, we have to lock the thread mutex here as well.
		EnterCriticalSection(&taskDesc->threadMutex);
		{
			// Create a thread to serve the task
			XME_CHECK_REC
			(
				NULL != (taskDesc->threadHandle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)&xme_hal_sched_taskProc, (LPVOID)newTaskHandle, 0, &taskDesc->threadId)),
				XME_HAL_SCHED_INVALID_TASK_HANDLE,
				{
					taskDesc->threadId = 0;
					LeaveCriticalSection(&taskDesc->threadMutex);
					if (NULL != taskDesc->taskPeriodTimer)
					{
						CloseHandle(taskDesc->taskPeriodTimer);
					}
					CloseHandle(taskDesc->stateChangedEvent);
					DeleteCriticalSection(&taskDesc->threadMutex);
					XME_HAL_TABLE_REMOVE_ITEM(xme_hal_sched_config.taskDescriptors, (xme_hal_table_rowHandle_t)newTaskHandle);
					LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
				}
			);

			// TODO: See ticket #813
			//SetThreadPriority(taskDesc->threadHandle, <prio>);
		}
		LeaveCriticalSection(&taskDesc->threadMutex);
	}
	LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);

	// Return the unique task handle
	return newTaskHandle;
}

xme_core_status_t
xme_hal_sched_setTaskExecutionState
(
	xme_hal_sched_taskHandle_t taskHandle,
	bool running
)
{
	xme_hal_sched_taskDescriptor_t* taskDesc;

	// Synchronize access to the task descriptors mutex
	EnterCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
	{
		// Verify the task handle
		XME_CHECK_REC
		(
			NULL != (taskDesc = XME_HAL_TABLE_ITEM_FROM_HANDLE(xme_hal_sched_config.taskDescriptors, taskHandle)),
			XME_CORE_STATUS_INVALID_HANDLE,
			{
				LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
			}
		);

		// Synchronize access to the thread mutex
		EnterCriticalSection(&taskDesc->threadMutex);
		{
			XME_CHECK_REC
			(
				(NULL != taskDesc->threadHandle) && (0 != taskDesc->threadId),
				XME_CORE_STATUS_INVALID_CONFIGURATION,
				{
					LeaveCriticalSection(&taskDesc->threadMutex);
					LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
				}
			);

			XME_CHECK_REC
			(
				XME_HAL_SCHED_TASK_STATE_TERMINATING != taskDesc->taskState,
				XME_CORE_STATUS_PERMISSION_DENIED,
				{
					LeaveCriticalSection(&taskDesc->threadMutex);
					LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
				}
			);

			// Check whether a state change is necessary
			if (!running ^ !!(taskDesc->taskState & XME_HAL_SCHED_TASK_STATE_SUSPENDED))
			{
				// Update suspended flag
				if (running)
				{
					taskDesc->taskState = XME_HAL_SCHED_TASK_STATE_RUNNING;
				}
				else
				{
					taskDesc->taskState = XME_HAL_SCHED_TASK_STATE_SUSPENDED;
				}

				// Signal the thread
				SetEvent(taskDesc->stateChangedEvent);
			}
		}
		LeaveCriticalSection(&taskDesc->threadMutex);
	}
	LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);

	return XME_CORE_STATUS_SUCCESS;
}

xme_core_status_t
xme_hal_sched_removeTask
(
	xme_hal_sched_taskHandle_t taskHandle
)
{
	xme_hal_sched_taskDescriptor_t* taskDesc;

	// Synchronize access to the task descriptors mutex
	EnterCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
	{
		bool selfCleanup = false;

		// Verify the task handle
		XME_CHECK_REC
		(
			NULL != (taskDesc = XME_HAL_TABLE_ITEM_FROM_HANDLE(xme_hal_sched_config.taskDescriptors, taskHandle)),
			XME_CORE_STATUS_INVALID_HANDLE,
			{
				LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
			}
		);

		// Synchronize access to the thread mutex
		EnterCriticalSection(&taskDesc->threadMutex);
		{
			XME_ASSERT((NULL != taskDesc->threadHandle) && (0 != taskDesc->threadId));

			// Schedule the task for termination (this will resume the task)
			taskDesc->taskState = XME_HAL_SCHED_TASK_STATE_TERMINATING;

			// If this function has been called from the context of the thread
			// to be removed, we set the selfCleanup flag and can safely
			// continue, as it is guaranteed that the thread will check the
			// task state before subsequently executing the callback function.
			// However, if this function has not been called from the context
			// of the thread, awake the thread, wait for it to terminate and
			// perform the cleanup ourselves to make sure the thread has
			// been terminated before the function returns.
			if (GetCurrentThreadId() == taskDesc->threadId)
			{
				taskDesc->selfCleanup = selfCleanup = true;
			}
			else
			{
				// TODO: Is this threadsafe? See ticket #753
				LeaveCriticalSection(&taskDesc->threadMutex);
				SignalObjectAndWait(taskDesc->stateChangedEvent, taskDesc->threadHandle, INFINITE, false);
				EnterCriticalSection(&taskDesc->threadMutex);

				// Close the thread handle
				CloseHandle(taskDesc->threadHandle);

				// Free resources allocated in the thread descriptor
				CloseHandle(taskDesc->stateChangedEvent);
			}
		}
		LeaveCriticalSection(&taskDesc->threadMutex);

		if (!selfCleanup)
		{
			// Free resources allocated in the task descriptor
			DeleteCriticalSection(&taskDesc->threadMutex);
			XME_HAL_TABLE_REMOVE_ITEM(xme_hal_sched_config.taskDescriptors, (xme_hal_table_rowHandle_t)taskHandle);
		}
	}
	LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);

	return XME_CORE_STATUS_SUCCESS;
}

void
xme_hal_sched_fini()
{
	// Synchronize access to the task descriptors mutex
	EnterCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
	{
		// Terminate all threads and wait for them to "join"
		XME_HAL_TABLE_ITERATE
		(
			xme_hal_sched_config.taskDescriptors,
			xme_hal_sched_taskHandle_t, taskHandle,
			xme_hal_sched_taskDescriptor_t, taskDesc,
			{
				xme_hal_sched_removeTask(taskHandle);
			}
		);
	}
	LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);

	// Destroy the task descriptors mutex
	DeleteCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);

	XME_HAL_TABLE_FINI(xme_hal_sched_config);
}

// TODO (See ticket #814): What about calling suspend, then immediately resume?
//                         Should the task be executed at now, now+period or at old+period?
//                         The current implementation is now. This means that calling suspend
//                         and resume eliminates the startup time given if it has not already
//                         elapsed.
DWORD
xme_hal_sched_taskProc
(
	LPVOID lpParameter
)
{
	xme_hal_sched_taskHandle_t taskHandle = (xme_hal_sched_taskHandle_t)lpParameter;
	xme_hal_sched_taskDescriptor_t* taskDesc;

	XME_ASSERT(0 != taskHandle);

	// Register the thread
	{
		xme_core_status_t result;

		result = xme_hal_random_registerThread();
		XME_ASSERT(XME_CORE_STATUS_SUCCESS == result);

		result = xme_hal_tls_registerThread();
		XME_ASSERT(XME_CORE_STATUS_SUCCESS == result);
	}

	// Retrieve the task context
	EnterCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
	{
		taskDesc = XME_HAL_TABLE_ITEM_FROM_HANDLE(xme_hal_sched_config.taskDescriptors, taskHandle);
	}
	LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);

	XME_ASSERT(NULL != taskDesc);
	XME_ASSERT
	(
		XME_HAL_SCHED_TASK_STATE_RUNNING == taskDesc->taskState ||
		XME_HAL_SCHED_TASK_STATE_SUSPENDED == taskDesc->taskState ||
		XME_HAL_SCHED_TASK_STATE_TERMINATING == taskDesc->taskState
	);

	// Synchronize access to the thread mutex
	EnterCriticalSection(&taskDesc->threadMutex);
	{
		xme_hal_time_interval_t waitTime = taskDesc->startMs;

		XME_ASSERT(NULL != taskDesc->callback);

		// Schedule the first invocation of the task
		// The first time to start the task might differ from the period.
		// This is why we specify the waitTime as initial relative due time
		// (given in 100 nanosecond units = 0.0001 milliseconds) and the
		// periodMs as regular period (in milliseconds).
		if (0 != waitTime && taskDesc->taskState != XME_HAL_SCHED_TASK_STATE_SUSPENDED)
		{
			LARGE_INTEGER dueTime;
			dueTime.QuadPart = -((LONGLONG)waitTime*10000);

			XME_CHECK_REC
			(
				0 != SetWaitableTimer(taskDesc->taskPeriodTimer, (const LARGE_INTEGER*)&dueTime, taskDesc->periodMs, NULL, NULL, false),
				ERROR_NOT_SUPPORTED,
				{
					LeaveCriticalSection(&taskDesc->threadMutex);
				}
			);
		}

		while (1)
		{
			while (0 != waitTime || taskDesc->taskState == XME_HAL_SCHED_TASK_STATE_SUSPENDED)
			{
				// The order of handles matters, since when both objects get signaled at
				// the same time, WaitForMultipleObjects() will return the index of the
				// first object signaled. Since the state of the task can always be
				// determined with the taskDesc->taskState member, it is meaningful to
				// put the waitable timer first.
				HANDLE handles[2] = { 0, 0 };
				DWORD numHandles;
				DWORD state;
				
				if (NULL == taskDesc->taskPeriodTimer)
				{
					handles[0] = taskDesc->stateChangedEvent;
					numHandles = 1;
				}
				else
				{
					handles[0] = taskDesc->taskPeriodTimer;
					handles[1] = taskDesc->stateChangedEvent;
					numHandles = 2;
				}

				// Check for task removal
				if (XME_HAL_SCHED_TASK_STATE_TERMINATING == taskDesc->taskState)
				{
					break;
				}

				// TODO (See ticket #753): Is this OK or is there a race condition involved?
				//                         Unfortunately we can't use condition variables when
				//                         we want to support Windows XP as well, for example.
				LeaveCriticalSection(&taskDesc->threadMutex);

				// This is required to give other threads the chance to acquire the thread mutex.
				// See ticket #476.
				SwitchToThread();

				state = WaitForMultipleObjects(numHandles, &handles[0], false, INFINITE);
				EnterCriticalSection(&taskDesc->threadMutex);

				// Check for task removal
				if (XME_HAL_SCHED_TASK_STATE_TERMINATING == taskDesc->taskState)
				{
					break;
				}
				else if (XME_HAL_SCHED_TASK_STATE_RUNNING == taskDesc->taskState)
				{
					if (0 == state - WAIT_OBJECT_0 && 2 == numHandles)
					{
						// The period timer has elapsed
						break;
					}

					// Assert that a state change has been signaled.
					// The following statement should not have any side effects,
					// because the event must be in signaled state.
					XME_ASSERT(XME_ASSERT_NO_SIDE_EFFECTS(WAIT_OBJECT_0 == WaitForSingleObject(taskDesc->stateChangedEvent, 0)));

					// Reset the event (set it to non-signaled state).
					// This is only done manually so that the above assertion can work.
					// If you do not need the assertion, set the bManualReset parameter
					// in the call to CreateEvent() in xme_hal_sched_addTask() to false.
					ResetEvent(taskDesc->stateChangedEvent);

					// The task has just been resumed. Schedule it for periodic execution.
					// The first time to start the task might differ from the period.
					// This is why we specify the waitTime as initial relative due time
					// (given in 100 nanosecond units = 0.0001 milliseconds) and the
					// periodMs as regular period (in milliseconds).
					if (NULL != taskDesc->taskPeriodTimer)
					{
						LARGE_INTEGER dueTime;
						dueTime.QuadPart = -((LONGLONG)waitTime*10000);

						XME_CHECK_REC
						(
							0 != SetWaitableTimer(taskDesc->taskPeriodTimer, (const LARGE_INTEGER*)&dueTime, taskDesc->periodMs, NULL, NULL, false),
							ERROR_NOT_SUPPORTED,
							{
								LeaveCriticalSection(&taskDesc->threadMutex);
							}
						);
					}

					// TODO (See ticket #814): We might want to execute the first iteration right now
					//                         instead of waiting for the taskPeriodTimer to become
					//                         signaled. However, this should only be done if it is
					//                         asserted that the task has been previously run and the
					//                         time between the last run and the current run is larger
					//                         than the period. For now, we just wait for the full time
					//                         to elapse, which means that a task will not run
					//                         immediately when resumed.
					//break;
				}
				else
				{
					ResetEvent(taskDesc->stateChangedEvent);

					// Cancel the waitable timer until the task is resumed
					CancelWaitableTimer(taskDesc->taskPeriodTimer);

					// The taskPeriodTimer should not be signaled when the task is not running!
					//XME_ASSERT_RVAL(0 != state - WAIT_OBJECT_0, ERROR_INTERNAL_ERROR);
				}
			}

			// Check for task removal
			if (XME_HAL_SCHED_TASK_STATE_TERMINATING == taskDesc->taskState)
			{
				if (NULL != taskDesc->taskPeriodTimer)
				{
					// Cancel the waitable timer
					CancelWaitableTimer(taskDesc->taskPeriodTimer);
				}

				break;
			}

			if (0 == waitTime && 0 != taskDesc->periodMs)
			{
				// The task has been run as quickly as possible.
				// We have to schedule the waitable timer before calling the
				// callback function for the first time.
				LARGE_INTEGER dueTime;
				dueTime.QuadPart = -((LONGLONG)taskDesc->periodMs*10000);

				XME_ASSERT(NULL != taskDesc->taskPeriodTimer);

				XME_CHECK_REC
				(
					0 != SetWaitableTimer(taskDesc->taskPeriodTimer, (const LARGE_INTEGER*)&dueTime, taskDesc->periodMs, NULL, NULL, false),
					ERROR_NOT_SUPPORTED,
					{
						LeaveCriticalSection(&taskDesc->threadMutex);
					}
				);
			}

			// Switch to the context of the component and run the task
			XME_COMPONENT_CONTEXT
			(
				taskDesc->componentContext,
				{
					taskDesc->callback(taskDesc->userData);
				}
			);

			// Remove the task if it was only scheduled for a single execution
			if (0 == taskDesc->periodMs)
			{
				xme_hal_sched_removeTask(taskHandle);
				break;
			}
			else
			{
				// The threadPeriodTimer must not be NULL for periodic tasks!
				XME_ASSERT(NULL != taskDesc->taskPeriodTimer);
			}

			// The waitTime is the period time for any subsequent resumes of the task
			waitTime = taskDesc->periodMs;
		}

		// Destroy the waitable timer
		if (NULL != taskDesc->taskPeriodTimer)
		{
			// Cancel the waitable timer
			CancelWaitableTimer(taskDesc->taskPeriodTimer);
			CloseHandle(taskDesc->taskPeriodTimer);
			taskDesc->taskPeriodTimer = NULL;
		}

		if (taskDesc->selfCleanup)
		{
			// Close the thread handle
			CloseHandle(taskDesc->threadHandle);

			// Free resources allocated in the thread descriptor
			CloseHandle(taskDesc->stateChangedEvent);

			// Leave the critical section
			LeaveCriticalSection(&taskDesc->threadMutex);

			// Free resources allocated in the task descriptor
			DeleteCriticalSection(&taskDesc->threadMutex);
			EnterCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);
			{
				XME_HAL_TABLE_REMOVE_ITEM(xme_hal_sched_config.taskDescriptors, (xme_hal_table_rowHandle_t)taskHandle);
			}
			LeaveCriticalSection(&xme_hal_sched_config.taskDescriptorsMutex);

			// Deregister the thread
			xme_hal_tls_deregisterThread();
			xme_hal_random_deregisterThread();

			return ERROR_SUCCESS;
		}
	}
	LeaveCriticalSection(&taskDesc->threadMutex);

	// Deregister the thread
	xme_hal_tls_deregisterThread();
	xme_hal_random_deregisterThread();

	return ERROR_SUCCESS;
}
