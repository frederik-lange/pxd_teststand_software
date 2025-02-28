/*
 * Copyright (c) 2011, fortiss GmbH.
 * All rights reserved.
 *
 * $Id$
 */

/**
 * \file
 *         Calibration Window
 *
 * \author
 *         Nadine Keddis <keddis@fortiss.org>
 */

 #include <QTest>
 #include <QSettings>
#include "calibrationwindow.h"
#include "ui_calibrationwindow.h"
#include "wrapperHeader.h"

#define CALIBRATE 2
#define RECALIBRATE 4


calibrationWindow::calibrationWindow(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::calibrationWindow)
{
    ui->setupUi(this);

	this->setFixedSize(this->width(), this->height());


	QString *path= new QString("C:/root/Callibration/");
	QString *iniFile=new QString("C:/root/Callibration/constants_node_");


    connect(ui->setValueVoltageButton, SIGNAL(clicked()), this, SLOT(setValueVoltage()));
	connect(ui->setCalibratedValueVoltageButton, SIGNAL(clicked()), this, SLOT(setCalibratedValueVoltage()));
	connect(ui->setValueCurrentButton, SIGNAL(clicked()), this, SLOT(setValueCurrent()));
	connect(ui->setCalibratedValueCurrentButton, SIGNAL(clicked()), this, SLOT(setCalibratedValueCurrent()));
    connect(ui->recalibrateButton, SIGNAL(clicked()), this, SLOT(recalibrate()));
	connect(ui->ResetCalPush, SIGNAL(clicked()), this, SLOT(ResetConstants()));
	connect(ui->LoadCalIniPush, SIGNAL(clicked()), this, SLOT(LoadConstants()));
	connect(ui->LoadChannelPush, SIGNAL(clicked()), this, SLOT(LoadChannel()));
}

calibrationWindow::~calibrationWindow()
{
    delete ui;
}

//This function is called when the Set Uncalibrated Voltage Value Button has been pressed
void calibrationWindow::setValueVoltage(){
    calibration_input_data_t data;
	data.nodeNumber = ui->calibrateNodeNumber->value();
    data.channelNumber = ui->calibrationChannel->currentIndex();
	data.calibrate_bool = NOT_CALIBRATE_VALUE;
	data.voltage_or_current_bool = SET_VOLTAGE;
    data.voltage = ui->voltage->value();
    data.current = ui->current->value();
    sendDataWrapper(CALIBRATE, &data, sizeof(data));
	updateInfoVoltage(data.nodeNumber, (int)data.channelNumber, (int)data.voltage);
}

//This function is called when the Set Calibrated Voltage Value Button has been pressed
void calibrationWindow::setCalibratedValueVoltage(){
    calibration_input_data_t data;
	data.nodeNumber = ui->calibrateNodeNumber->value();
    data.channelNumber = ui->calibrationChannel->currentIndex();
	data.calibrate_bool = CALIBRATE_VALUE;
	data.voltage_or_current_bool = SET_VOLTAGE;
    data.voltage = ui->voltage->value();
    data.current = ui->current->value();
    sendDataWrapper(CALIBRATE, &data, sizeof(data));
	updateInfoVoltage(data.nodeNumber, (int)data.channelNumber, (int)data.voltage);
}

//This function is called when the Set Uncalibrated Current Value Button has been pressed
void calibrationWindow::setValueCurrent(){
    calibration_input_data_t data;
	data.nodeNumber = ui->calibrateNodeNumber->value();
    data.channelNumber = ui->calibrationChannel->currentIndex();
	data.calibrate_bool = NOT_CALIBRATE_VALUE;
	data.voltage_or_current_bool = SET_CURRENT;
    data.voltage = ui->voltage->value();
    data.current = ui->current->value();
    sendDataWrapper(CALIBRATE, &data, sizeof(data));
	updateInfoCurrent(data.nodeNumber, (int)data.channelNumber, data.current);
}

//This function is called when the Set Calibrated Current Value Button has been pressed
void calibrationWindow::setCalibratedValueCurrent(){
    calibration_input_data_t data;
	data.nodeNumber = ui->calibrateNodeNumber->value();
    data.channelNumber = ui->calibrationChannel->currentIndex();
	data.calibrate_bool = CALIBRATE_VALUE;
	data.voltage_or_current_bool = SET_CURRENT;
    data.voltage = ui->voltage->value();
    data.current = ui->current->value();
    sendDataWrapper(CALIBRATE, &data, sizeof(data));
	updateInfoCurrent(data.nodeNumber, (int)data.channelNumber, data.current);
}

//This function is called when the Recalibrate Button has been pressed
void calibrationWindow::recalibrate(){
    recalibration_data_t data;
	calibration_channel_t channel;
	data.nodeNumber = ui->calibrateNodeNumber->value();
	data.channelNumber = ui->calibrationChannel->currentIndex();
	channel.gain = ui->dacU_gain->value();
	channel.offset = ui->dacU_offset->value();
	data.calibration_page[0] = channel;
	channel.gain = ui->dacI_gain->value();
	channel.offset = ui->dacI_offset->value();
	data.calibration_page[1] = channel;
	channel.gain = ui->adcI_gain->value();
	channel.offset = ui->adcI_offset->value();
	data.calibration_page[2] = channel;
	channel.gain = ui->adcU_out_gain->value();
	channel.offset = ui->adcU_out_offset->value();
	data.calibration_page[3] = channel;
	channel.gain = ui->adcU_load_gain->value();
	channel.offset = ui->adcU_load_offset->value();
	data.calibration_page[4] = channel; 
    sendDataWrapper(RECALIBRATE, &data, sizeof(data));
}



void calibrationWindow::ResetConstants(){

	recalibration_data_t data;
	calibration_channel_t channel;
	data.nodeNumber = ui->calibrateNodeNumber->value();
	
	QString message="";

for(int i=0;i<24;i++){

	data.channelNumber = i;
	channel.gain = 10000;
	channel.offset = 0;
	data.calibration_page[0] = channel;
	channel.gain = 10000;
	channel.offset = 0;
	data.calibration_page[1] = channel;
	channel.gain = 10000;
	channel.offset = 0;
	data.calibration_page[2] = channel;
	channel.gain = 10000;
	channel.offset = 0;
	data.calibration_page[3] = channel;
	channel.gain = 10000;
	channel.offset = 0;
	data.calibration_page[4] = channel; 
    sendDataWrapper(RECALIBRATE, &data, sizeof(data));
	QTest::qWait(626);

	message.append(" Reset Channel ");
	message.append(QString::number(i));

	emit SendInfo(message);



}


}

void calibrationWindow::LoadChannel(){

QString path="C:/root/Callibration/";
QString iniFile="C:/root/Callibration/constants_node_";

iniFile.append(QString::number(ui->calibrateNodeNumber->value()));
iniFile.append(".ini");

QString message;

message="Load: ";
message.append(iniFile);

emit SendInfo(message);

double offset,gain;

QSettings *settings = new QSettings(iniFile,QSettings::IniFormat); //QSettings::UserScope);
			settings->setDefaultFormat(QSettings::IniFormat);
			settings->setPath(QSettings::IniFormat,QSettings::UserScope,path);
			settings->sync();

	recalibration_data_t data;
	calibration_channel_t channel;
	data.nodeNumber = ui->calibrateNodeNumber->value();

int sel_ch=ui->calibrationChannel->currentIndex();

	QString tag_string1="";
	QString tag_string2=""; 


/////////////////////////////////////////////////////////////////////
    tag_string1="";
	tag_string1.append(QString::number(sel_ch));
	tag_string1.append("/DAC_VOLTAGE_GAIN");
	
	tag_string2="";
	tag_string2.append(QString::number(sel_ch));
	tag_string2.append("/DAC_VOLTAGE_OFFSET");

	message="Channel ";
	message.append(QString::number(sel_ch));

	data.channelNumber = sel_ch;
	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	 ui->dacU_gain->setValue((int)gain);
	 ui->dacI_offset->setValue((int)offset);
	
	//channel.offset = (int)offset;
	//data.calibration_page[0] = channel;
	


/////////////////////////////////////////////////////////////////////	

	tag_string1="";
	tag_string1.append(QString::number(sel_ch));
	tag_string1.append("/ADC_U_REGULATOR_GAIN");
	
	tag_string2="";
	tag_string2.append(QString::number(sel_ch));
	tag_string2.append("/ADC_U_REGULATOR_OFFSET");
	
	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	
	 ui->adcU_out_gain->setValue((int)gain);
	 ui->adcU_out_offset->setValue((int)offset);

	 channel.gain=(int)gain;
	channel.offset = (int)offset;

	data.calibration_page[1] = channel;
	


/////////////////////////////////////////////////////////////////////
	tag_string1="";
	tag_string1.append(QString::number(sel_ch));
	tag_string1.append("/ADC_U_LOAD_GAIN");
	

	tag_string2="";
	tag_string2.append(QString::number(sel_ch));
	tag_string2.append("/ADC_U_LOAD_OFFSET");

	
	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	
	ui->adcU_load_gain->setValue((int)gain);
	ui->adcU_load_offset->setValue((int)offset);
	 
	 channel.gain=(int)gain;
	channel.offset = (int)offset;
	data.calibration_page[2] = channel;
	


/////////////////////////////////////////////////////////////////////
	 tag_string1="";
	tag_string1.append(QString::number(sel_ch));
	tag_string1.append("/ADC_I_MON_GAIN");

	tag_string2="";
	tag_string2.append(QString::number(sel_ch));
	tag_string2.append("/ADC_I_MON_OFFSET");

	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	if(gain<=0) gain=gain*(-1.0);
	 channel.gain=(int)gain;
	channel.offset =offset; // (int)offset;
	
	ui->adcI_gain->setValue((int)gain);
	ui->adcI_offset->setValue((int)offset);

	data.calibration_page[3] = channel;

/////////////////////////////////////////////////////////////////////
	tag_string1="";
	tag_string1.append(QString::number(sel_ch));
	tag_string1.append("/DAC_CURRENT_GAIN");
	
	tag_string2="";
	tag_string2.append(QString::number(sel_ch));
	tag_string2.append("/DAC_CURRENT_OFFSET");

	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	if(gain<=0) gain=gain*(-1.0);
	 channel.gain=(int)gain;
	channel.offset =0; // (int)offset;
	data.calibration_page[4] = channel; 

	ui->dacI_gain->setValue((int)gain);
	ui->dacU_offset->setValue((int)offset);

//	sendDataWrapper(RECALIBRATE, &data, sizeof(data));
	


	



}

void calibrationWindow::LoadConstants(){

QString path="C:/root/Callibration/";
QString iniFile="C:/root/Callibration/constants_node_";

iniFile.append(QString::number(ui->calibrateNodeNumber->value()));
iniFile.append(".ini");

QString message;

message="Load: ";
message.append(iniFile);

emit SendInfo(message);

double offset,gain;

QSettings *settings = new QSettings(iniFile,QSettings::IniFormat); //QSettings::UserScope);
			settings->setDefaultFormat(QSettings::IniFormat);
			settings->setPath(QSettings::IniFormat,QSettings::UserScope,path);
			settings->sync();

	recalibration_data_t data;
	calibration_channel_t channel;
	data.nodeNumber = ui->calibrateNodeNumber->value();

	QString tag_string1="";
	QString tag_string2=""; 

	for(int i=0;i<24;i++){

/////////////////////////////////////////////////////////////////////
    tag_string1="";
	tag_string1.append(QString::number(i));
	tag_string1.append("/DAC_VOLTAGE_GAIN");
	
	tag_string2="";
	tag_string2.append(QString::number(i));
	tag_string2.append("/DAC_VOLTAGE_OFFSET");

	message="Channel: \n ";
	message.append(QString::number(i));

	data.channelNumber = i;
	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	channel.gain=(int)gain;
	channel.offset = offset; // (int)offset;
	data.calibration_page[0] = channel;
	
	message.append(" Dac gain: ");
	message.append(QString::number(channel.gain));
	message.append(" Dac offset: ");
	message.append(QString::number(channel.offset));

/////////////////////////////////////////////////////////////////////
	tag_string1="";
	tag_string1.append(QString::number(i));
	tag_string1.append("/DAC_CURRENT_GAIN");
	
	tag_string2="";
	tag_string2.append(QString::number(i));
	tag_string2.append("/DAC_CURRENT_OFFSET");

	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	if(gain<=0) gain=gain*(-1.0);
	 channel.gain=(int)gain;
	channel.offset =offset; // (int)offset;
	data.calibration_page[1] = channel; 
    message.append(" DAC_I G: ");
	message.append(QString::number(channel.gain));
	message.append(" DAC_I O: ");
	message.append(QString::number(channel.offset));
	message.append("\n");
/////////////////////////////////////////////////////////////////////
	 tag_string1="";
	tag_string1.append(QString::number(i));
	tag_string1.append("/ADC_I_MON_GAIN");

	tag_string2="";
	tag_string2.append(QString::number(i));
	tag_string2.append("/ADC_I_MON_OFFSET");

	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	if(gain<=0) gain=gain*(-1.0);
	 channel.gain=(int)gain;
	channel.offset =offset; // (int)offset;
	 
	data.calibration_page[2] = channel;
	
	message.append(" MON_I G: ");
	message.append(QString::number(channel.gain));
	message.append(" MON_I Ooff: ");
	message.append(QString::number(channel.offset));
/////////////////////////////////////////////////////////////////////	

	/*tag_string1="";
	tag_string1.append(QString::number(i));
	tag_string1.append("/ADC_U_REGULATOR_GAIN");
	
	tag_string2="";
	tag_string2.append(QString::number(i));
	tag_string2.append("/ADC_U_REGULATOR_OFFSET"); */

	tag_string1="";
	tag_string1.append(QString::number(i));
	tag_string1.append("/ADC_U_LOAD_GAIN");
	

	tag_string2="";
	tag_string2.append(QString::number(i));
	tag_string2.append("/ADC_U_LOAD_OFFSET"); 
	
	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	channel.gain=(int)gain;
	channel.offset = (int)offset; //offset;

	data.calibration_page[3] = channel;
	
	message.append(" ADC_Reg G: ");
	message.append(QString::number(channel.gain));
	message.append(" ADC_reg Off: ");
	message.append(QString::number(channel.offset));
	message.append("\n");

/////////////////////////////////////////////////////////////////////
/*	tag_string1="";
	tag_string1.append(QString::number(i));
	tag_string1.append("/ADC_U_LOAD_GAIN");
	

	tag_string2="";
	tag_string2.append(QString::number(i));
	tag_string2.append("/ADC_U_LOAD_OFFSET"); */

	tag_string1="";
	tag_string1.append(QString::number(i));
	tag_string1.append("/ADC_U_REGULATOR_GAIN");
	
	tag_string2="";
	tag_string2.append(QString::number(i));
	tag_string2.append("/ADC_U_REGULATOR_OFFSET");

	
	 gain= settings->value(tag_string1.toAscii().data(),10000).toDouble();
	 offset=settings->value(tag_string2.toAscii().data(),0).toDouble();
	channel.gain=(int)gain;
	channel.offset = offset; //(int)offset;
	data.calibration_page[4] = channel;
	
	message.append(" ADC_Load G: ");
	message.append(QString::number(channel.gain));
		message.append(" ADC_Load Off: ");
	message.append(QString::number(channel.offset));



	sendDataWrapper(RECALIBRATE, &data, sizeof(data));
	
	QTest::qWait(726);
	QTest::qSleep(226);
	emit SendInfo(message);
	
	QTest::qWait(106);
}





}