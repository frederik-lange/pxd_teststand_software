/*
 * Copyright (c) 2011, fortiss GmbH.
 * All rights reserved.
 *
 * $Id$
 */

/**
 * \file
 *         GUI Header
 *
 * \author
 *         Nadine Keddis <keddis@fortiss.org>
 */
#ifdef __cplusplus
extern "C" { 
#endif
	int startGui(int argc, char *argv[]);
	void displayNodeChannel(int node, int channel);
	void displayLogdata(int node, char *data, int number);
	void updateStatus(int node, int status);
	void updateMonitor(int node, monitor_information_node_t info);
	void updateLogFile(int node, char *data, int number);
#ifdef __cplusplus
}
#endif
void sendDataWrapper(int topic, void* data, unsigned short size);
void shutdownWrapper();
void startupWrapper();
