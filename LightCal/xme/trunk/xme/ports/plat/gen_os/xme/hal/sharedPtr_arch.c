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
 *         Shared memory abstraction (architecture specific part: generic OS
 *         based implementation).
 *
 * \author
 *         Michael Geisinger <geisinger@fortiss.org>
 *         Stephan Sommer <sommer@fortiss.org>
 */

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "xme/hal/sharedPtr.h"
#include "xme/hal/mem.h"

#include "xme/core/defines.h"

#include <stdlib.h>

/******************************************************************************/
/***   Static variables                                                     ***/
/******************************************************************************/
xme_hal_sharedPtr_configStruct_t xme_hal_sharedPtr_config;

/******************************************************************************/
/***   Implementation                                                       ***/
/******************************************************************************/
// TODO: Inline all these functions on this platform! See ticket #822
xme_core_status_t
xme_hal_sharedPtr_init()
{
	XME_CHECK
	(
		XME_HAL_SYNC_INVALID_CRITICAL_SECTION_HANDLE !=
		(
			xme_hal_sharedPtr_config.criticalSectionHandle = xme_hal_sync_createCriticalSection()
		),
		XME_CORE_STATUS_OUT_OF_RESOURCES
	);

	return XME_CORE_STATUS_SUCCESS;
}

void
xme_hal_sharedPtr_fini()
{
	xme_hal_sync_destroyCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);
	xme_hal_sharedPtr_config.criticalSectionHandle = XME_HAL_SYNC_INVALID_CRITICAL_SECTION_HANDLE;
}
/*
void*
xme_hal_mem_alloc
(
	uint16_t size
)
{
	return malloc(size);
}

void*
xme_hal_mem_realloc
(
	void* buf,
	uint16_t size
)
{
	return realloc(buf, size);
}

void
xme_hal_mem_free
(
	void* memory
)
{
	free(memory);
}
*/
xme_hal_sharedPtr_bufferRow_t xme_hal_sharedPtr_buffer[XME_CORE_BROKER_PACKET_BUFFER];


xme_hal_sharedPtr_t
xme_hal_sharedPtr_create
(
	uint16_t size
)
{
	xme_hal_sharedPtr_t sharedPtr;

	xme_hal_sync_enterCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);
	{
		for (sharedPtr = (xme_hal_sharedPtr_t)0; sharedPtr < XME_CORE_BROKER_PACKET_BUFFER; sharedPtr = (xme_hal_sharedPtr_t)(((uint16_t)sharedPtr)+1))
		{
			if (0 == xme_hal_sharedPtr_buffer[sharedPtr].referenceCount)
			{
				if (NULL == (xme_hal_sharedPtr_buffer[sharedPtr].slot_data = xme_hal_mem_alloc(size)))
				{
					// Out of memory
					break;
				}

				xme_hal_sharedPtr_buffer[sharedPtr].referenceCount = 1;
				xme_hal_sharedPtr_buffer[sharedPtr].size = size;

				xme_hal_sync_leaveCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);

				return (xme_hal_sharedPtr_t)(((uint16_t)sharedPtr)+1);
			}
		}
	}
	xme_hal_sync_leaveCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);

	return XME_HAL_SHAREDPTR_INVALID_POINTER;
}

xme_hal_sharedPtr_t
xme_hal_sharedPtr_realloc
(
	xme_hal_sharedPtr_t sharedPtr,
	uint16_t size
)
{
	xme_hal_sharedPtr_bufferRow_t* row;

	// If the given buffer handle is invalid, allocate a new memory block
	// and return a handle to it
	if (XME_HAL_SHAREDPTR_INVALID_POINTER == sharedPtr)
	{
		return xme_hal_sharedPtr_create(size);
	}

	XME_ASSERT_RVAL(sharedPtr < XME_CORE_BROKER_PACKET_BUFFER, XME_HAL_SHAREDPTR_INVALID_POINTER);
	
	xme_hal_sync_enterCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);
	{
		// TODO: row can never be NULL, because xme_hal_sharedPtr_buffer is a static array! See ticket #854
		row = &xme_hal_sharedPtr_buffer[sharedPtr-1];
		XME_CHECK_REC(NULL != row, XME_HAL_SHAREDPTR_INVALID_POINTER, xme_hal_sync_leaveCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle));

		if (NULL != (row->slot_data = xme_hal_mem_realloc(row->slot_data, size)))
		{
			row->size = size;
		} // TODO: else? See ticket #856
	}
	xme_hal_sync_leaveCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);

	return sharedPtr;
}

xme_hal_sharedPtr_t
xme_hal_sharedPtr_retain
(
	xme_hal_sharedPtr_t sharedPtr
)
{
	XME_CHECK
	(
		XME_HAL_SHAREDPTR_INVALID_POINTER != sharedPtr,
		XME_HAL_SHAREDPTR_INVALID_POINTER
	);

	xme_hal_sync_enterCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);
	{
		XME_CHECK_REC
		(
			xme_hal_sharedPtr_buffer[sharedPtr-1].referenceCount > 0,
			XME_HAL_SHAREDPTR_INVALID_POINTER,
			{
				xme_hal_sync_leaveCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);
			}
		);

		xme_hal_sharedPtr_buffer[sharedPtr-1].referenceCount++;
	}
	xme_hal_sync_leaveCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);

	return sharedPtr;
}

void
xme_hal_sharedPtr_destroy
(
	xme_hal_sharedPtr_t sharedPtr
)
{
	XME_CHECK
	(
		XME_HAL_SHAREDPTR_INVALID_POINTER != sharedPtr,
	);

	xme_hal_sync_enterCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);
	{
		XME_ASSERT_NORVAL(xme_hal_sharedPtr_buffer[sharedPtr-1].referenceCount > 0);

		xme_hal_sharedPtr_buffer[sharedPtr-1].referenceCount--;
		if (xme_hal_sharedPtr_buffer[sharedPtr-1].referenceCount == 0)
		{
			xme_hal_mem_free(xme_hal_sharedPtr_buffer[sharedPtr-1].slot_data);
			xme_hal_sharedPtr_buffer[sharedPtr-1].size = 0;
		}
	}
	xme_hal_sync_leaveCriticalSection(xme_hal_sharedPtr_config.criticalSectionHandle);
}

xme_hal_sharedPtr_t
xme_hal_sharedPtr_createFromPointer
(
	uint16_t size,
	void *data
)
{
	xme_hal_sharedPtr_t sharedPtr;

	if (XME_HAL_SHAREDPTR_INVALID_POINTER != (sharedPtr = xme_hal_sharedPtr_create(size)))
	{
		memcpy((xme_hal_sharedPtr_buffer[sharedPtr-1].slot_data), data, size);

		return sharedPtr;
	}
	
	return XME_HAL_SHAREDPTR_INVALID_POINTER;
}

uint16_t
xme_hal_sharedPtr_getSize
(
	xme_hal_sharedPtr_t sharedPtr
)
{
	return xme_hal_sharedPtr_buffer[sharedPtr-1].size;
};



xme_hal_sharedPtr_referenceCount_t
xme_hal_sharedPtr_getReferenceCount
(
	xme_hal_sharedPtr_t sharedPtr
)
{
	if (xme_hal_sharedPtr_buffer[sharedPtr-1].referenceCount > 0)
	{
		return xme_hal_sharedPtr_buffer[sharedPtr-1].referenceCount;
	}
	else
	{
		return XME_HAL_SHAREDPTR_INVALID_POINTER;
	}
}


void*
xme_hal_sharedPtr_getPointer
(
	xme_hal_sharedPtr_t sharedPtr
)
{
	return xme_hal_sharedPtr_buffer[sharedPtr-1].slot_data;
}
