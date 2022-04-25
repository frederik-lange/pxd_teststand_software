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
 *         This is the stub of a test component, based on the SAFE design. It
 *         can be used to create new test components very easily by making a
 *         copy of it and changing the sections, marked with 'TODO'. In addition,
 *         it might be necessary to extend xme_adv_testStub_configStruct_t, if
 *         additional information is required to execute the test. Please have
 *         a look at the heartbeatComponent and the heartbeatTestComponent for
 *         an example how tests with external evidence generators can be imple-
 *         mented.
 *
 * \author
 *         Dominik Sojer <sojer@fortiss.org>
 */

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "testStub.h"

/******************************************************************************/
/***   Static variables                                                     ***/
/******************************************************************************/

/******************************************************************************/
/***   Implementation                                                       ***/
/******************************************************************************/


xme_core_status_t xme_adv_testStub_create(xme_adv_testStub_configStruct_t* config)
{
	// init instant error reaction
	errorReaction = NULL;
	errorReaction = config->callback;

	// Create publisher for error message topic
	config->pubHandle = xme_core_dcc_publishTopic( 
		ERROR_MESSAGE_TOPIC_ID, 
		XME_CORE_MD_EMPTY_META_DATA, 
		false,
		NULL
	);

	if ( pubHandle == XME_CORE_DCC_INVALID_PUBLICATION_HANDLE )
	{
		return XME_CORE_STATUS_INTERNAL_ERROR;
	} 

	// Local channel from testStub component
	if ( xme_core_routingTable_addLocalSourceRoute( 
		ERROR_MESSAGE_CHANNEL, 
		xme_core_dcc_getComponentFromPublicationHandle(pubHandle),
		xme_core_dcc_getPortFromPublicationHandle(pubHandle) 
	) != XME_CORE_STATUS_SUCCESS ) 	
	{
		return XME_CORE_STATUS_INTERNAL_ERROR;
	}
	
	return XME_CORE_STATUS_SUCCESS;
}

xme_core_status_t xme_adv_testStub_activate(xme_adv_testStub_configStruct_t* config)
{
	xme_adv_healthmonitor_errorMessage_t error;

	xme_hal_sched_addTask(config->interval_ms, config->interval_ms, 0, test_callback, NULL);

	// notification that test is running
	error.componentId = xme_core_resourceManager_getCurrentComponentId();
	error.nodeId = xme_core_nodeManager_getNodeId();
	error.status = XME_ADV_HEALTHMONITOR_TEST_OK;
	error.identifier = XME_ADV_TEST_CPU; // TODO: change identifier for specific test
	xme_core_dcc_sendTopicData(pubHandle, &error, sizeof(error));

	return XME_CORE_STATUS_SUCCESS;
}

void xme_adv_testStub_deactivate(xme_adv_testStub_configStruct_t* config)
{
}

void xme_adv_testStub_destroy(xme_adv_testStub_configStruct_t* config) {
	if ( xme_core_dcc_unpublishTopic(pubHandle) != XME_CORE_STATUS_SUCCESS )
	{
		XME_LOG(XME_LOG_WARNING, "Failure xme_core_dcc_unsubscribeTopic\n");
	}
}

void test_callback (void* userData)
{
	xme_adv_testStub_configStruct_t* config = (xme_adv_testStub_configStruct_t*)userData;
	xme_adv_healthmonitor_errorMessage_t error;
	xme_core_status_t testResult;

	/* TODO: implement specific test function. Set 'testResult' to
	XME_CORE_STATUS_SUCCESS for successful tests and set it to 
	XME_CORE_STATUS_INTERNAL_ERROR if an error as been detected. */

	if (testResult != XME_CORE_STATUS_SUCCESS)
	{
		error.componentId = xme_core_resourceManager_getCurrentComponentId();
		error.nodeId = xme_core_nodeManager_getNodeId();
		error.status = XME_ADV_HEALTHMONITOR_TEST_FAILED;
		error.identifier = XME_ADV_TEST_MEMORY; // TODO: change identifier for specific test
		xme_core_dcc_sendTopicData(config->pubHandle, &error, sizeof(error));

		// instant error reaction
		if (config->callback != NULL) 
		{
			config->callback();
		} else {
			printf("error function not defined\n");
		}
	}
}
