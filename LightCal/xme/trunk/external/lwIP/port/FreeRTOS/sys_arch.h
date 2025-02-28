/*
 * Copyright (c) 2001-2003 Swedish Institute of Computer Science.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 * This file is part of the lwIP TCP/IP stack.
 *
 * Author: Adam Dunkels <adam@sics.se>
 *
 */
#ifndef __SYS_RTXC_H__
#define __SYS_RTXC_H__

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"

//----------------------------------------------------------------------------
//- Defines
//----------------------------------------------------------------------------
#define SYS_DEFAULT_THREAD_STACK_DEPTH	configMINIMAL_STACK_SIZE

// This will map mutexes to binary semaphores. It is not part of the official
// "opt.h" file of lwIP, so we define it here and won't include it into the
// cmake system.
#define LWIP_COMPAT_MUTEX 1

// lwIP needs a function returning a random 32-bit value
#define LWIP_RAND() lwip_rand_arch()

/* Message queue constants. */
#define archMESG_QUEUE_LENGTH	( 6 )

#define INT_MAX 32767
//----------------------------------------------------------------------------
//- Prototypes
//----------------------------------------------------------------------------
u32_t lwip_rand_arch(void);

//----------------------------------------------------------------------------
//- Declarations / typedefs
//----------------------------------------------------------------------------

typedef struct {
	xSemaphoreHandle handle;
	char valid;
} sys_sem_t;

typedef struct {
	xQueueHandle handle;
	char valid;
} sys_mbox_t;

typedef struct _sys_arch_state_t
{
	// Task creation data.
	char cTaskName[configMAX_TASK_NAME_LEN];
	unsigned short nStackDepth;
	unsigned short nTaskCount;
} sys_arch_state_t;

typedef xTaskHandle sys_thread_t;

#endif /* __SYS_RTXC_H__ */

