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
 *         Consistency test component.
 *
 * \author
 *         Dominik Sojer <sojer@fortiss.org>
 */

/******************************************************************************/
/***   Includes                                                             ***/
/******************************************************************************/
#include "testConsistency.h"

/******************************************************************************/
/***   Static variables                                                     ***/
/******************************************************************************/

static xme_core_dcc_publicationHandle_t pubHandle; // error messages

XME_HAL_TABLE
(
	checkFunctionPointer_t,
	xme_adv_testConsistency_functionsTable,
	XME_ADV_TESTCONSISTENCY_FUNCTIONS_MAX
);

/******************************************************************************/
/***   Implementation                                                       ***/
/******************************************************************************/

xme_core_status_t xme_adv_testConsistency_create(xme_adv_testConsistency_configStruct_t* config)
{
	XME_HAL_TABLE_INIT( xme_adv_testConsistency_functionsTable );
	initFunctionsTable();

	// Create publisher for error message topic
	pubHandle = xme_core_dcc_publishTopic( 
		ERROR_MESSAGE_TOPIC_ID, 
		XME_CORE_MD_EMPTY_META_DATA, 
		false,
		NULL
	);

	if ( pubHandle == XME_CORE_DCC_INVALID_PUBLICATION_HANDLE )
	{
		return XME_CORE_STATUS_INTERNAL_ERROR;
	} 

	// Local channel from testConsistency component
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

xme_core_status_t xme_adv_testConsistency_activate(xme_adv_testConsistency_configStruct_t* config)
{
	xme_adv_healthmonitor_errorMessage_t error;

	xme_hal_sched_addTask(config->startMs, config->periodMs, 0, testConsistency_callback, NULL);

	// notification that test is running
	error.componentId = xme_core_resourceManager_getCurrentComponentId();
	error.nodeId = xme_core_nodeManager_getNodeId();
	error.status = XME_ADV_HEALTHMONITOR_TEST_OK;
	error.identifier = XME_ADV_TEST_CONSISTENCY;
	xme_core_dcc_sendTopicData(pubHandle, &error, sizeof(error));

	return XME_CORE_STATUS_SUCCESS;
}

void xme_adv_testConsistency_deactivate(xme_adv_testConsistency_configStruct_t* config)
{
}

void xme_adv_testConsistency_destroy(xme_adv_testConsistency_configStruct_t* config) {
	if ( xme_core_dcc_unpublishTopic(pubHandle) != XME_CORE_STATUS_SUCCESS )
	{
		XME_LOG(XME_LOG_WARNING, "Failure xme_core_dcc_unsubscribeTopic\n");
	}
}

void testConsistency_callback (void* userData)
{
	xme_adv_healthmonitor_errorMessage_t error;
	checkFunctionPointer_t tempFunc;
	int failedTestsCount = 0;

	// execute consistency checks
	XME_HAL_TABLE_ITERATE
	(
		xme_adv_testConsistency_functionsTable,
		xme_hal_table_rowHandle_t, functionsHandle,
		checkFunctionPointer_t, function,
		{
			tempFunc = *function;
			if (tempFunc() == false)
			{
				failedTestsCount++;
			}
		}
	);

	if (failedTestsCount > 0) {
		error.componentId = xme_core_resourceManager_getCurrentComponentId();
		error.nodeId = xme_core_nodeManager_getNodeId();
		error.status = XME_ADV_HEALTHMONITOR_TEST_FAILED;
		error.identifier = XME_ADV_TEST_CONSISTENCY;
		xme_core_dcc_sendTopicData(pubHandle, &error, sizeof(error)); // send error message

		XME_LOG(XME_LOG_ERROR, "XME_ADV_HEALTHMONITOR_TEST_FAILED for node %d -> XME_ADV_TEST_CONSISTENCY\n", error.nodeId);
	} 
}

void initFunctionsTable() {
	xme_hal_table_rowHandle_t handle;
	checkFunctionPointer_t* inRow;

	handle = XME_HAL_TABLE_ADD_ITEM(xme_adv_testConsistency_functionsTable);
	inRow = XME_HAL_TABLE_ITEM_FROM_HANDLE( xme_adv_testConsistency_functionsTable, handle );
	*inRow = (checkFunctionPointer_t)func1;

	handle = XME_HAL_TABLE_ADD_ITEM(xme_adv_testConsistency_functionsTable);
	inRow = XME_HAL_TABLE_ITEM_FROM_HANDLE( xme_adv_testConsistency_functionsTable, handle );
	*inRow = (checkFunctionPointer_t)func2;
}

bool func1() {
	XME_LOG(XME_LOG_ERROR, "FUNC 1\n");
	XME_LOG(XME_LOG_ERROR, "FUNC 1\n");
	XME_LOG(XME_LOG_ERROR, "FUNC 1\n");
	return true;
}

bool func2() {
	XME_LOG(XME_LOG_ERROR, "FUNC 2\n");
	XME_LOG(XME_LOG_ERROR, "FUNC 2\n");
	XME_LOG(XME_LOG_ERROR, "FUNC 2\n");
	return false;
}