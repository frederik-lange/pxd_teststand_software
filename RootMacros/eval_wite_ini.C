{

//#include <iostream>
//#include <format>
//#include <fstream>
#include <sstream>
#include "TString.h"

//********************************************************
// some definitions
//********************************************************

cout<<"Definitions:"<< endl;
gROOT->Reset();
gROOT->ProcessLine(".L bisec.C");
gROOT->ProcessLine(".L graphCleaner.C");
int num_files=24; // fixed value! --> files have ID, files not there are set to default
//int num_cal=5;   // not used
int node=87;
int max_point=1000;
TString pathname;

bool data_available[24];

// fitting stuff
double frac,range;
double upper_limit; 
double lower_limit; 
int index_up,index_low;
double range_low,range_high;
double x,y;
double comp_x_min,comp_x_max,comp_y_min,comp_y_max;
int rem;

TString tags[8];

tags[0]="DAC_VOLTAGE_";
tags[1]="ADC_U_LOAD_";
tags[2]="ADC_U_REGULATOR_";
tags[3]="ADC_I_MON_";
tags[4]="DAC_CURRENT_";



TString offset="OFFSET";
TString gain  ="GAIN";


int num_pts;



//********************************************************
cout<<"Define Path to Data:"<< endl;

TString path="/home/silab44/pxd_teststand_software_frederik/data/example/";

// this is the default path the calibration outputs to
path="/home/silab44/pxd_teststand_software_frederik/data/example/";

// all the following pathes are for debugging only
// path = "C:/root/callibration/test2_PS41_FS/";
// testing only
// path="C:/root/callibration/test_PS41_fastShutdown_oldMCU/";
// path="C:/root/callibration/test_PS41_reworked_frontPCBv3_2/";
//path="C:/root/callibration/MainProdNode49/";#


//path="C:/root/Callibration/PreprodNODE";
//path="C:/root/Callibration/MainProdNODE";
//path+=node;
//path+="/";
//path="C:/root/Callibration/MainProdNode58_new/";
//path="C:/root/Callibration/MainProdNode58_withFastShutDown/";

cout<<"tags and path parts"<< endl;
TString part1="Channel_";
TString part3=".dat";
TString uu="_U_vs_U";
TString ii="_I_vs_I";
TString ilim="_Ilimit_vs_I";

TString string;
TGraph *graph[6];
TGraph *res_graph[6];
TGraph *graph_help[6];
TF1 *lin[7];

double resi;
double rel_resi;
double param[24][7][2];
double parameter2[2];

bool fall[5];
fall[0]=false;
fall[1]=false;
fall[2]=false;
fall[3]=true;
fall[4]=true;

TCanvas *tcan[24];

TString can;
bool falling;
ifstream fin;
double placeholder,place_holder;

for( int i=0; i<num_files;i++){ //num_files;i++){

// i is channel number

string=path;
string+=part1;
string+=i;
string+=uu;
string+=part3;

cout<<"Preparing Graphs:"<<string.Data()<<endl;

data_available[i]=false;

fin.open(string.Data());
    if (fin)  // check to see if file exists
    {
      data_available[i]=true;
    }
    fin.close();
  

if( data_available[i]==true){

	can="Channel";
	can+=i;
	tcan[i]=new TCanvas(can.Data(),can.Data());
	tcan[i]->Divide(5,3);


	graph[0]= new TGraph(string.Data(),"%lg %lg %*lg %*lg %*lg");  // * means skipping
	
	
	
	tcan[i]->cd(1);
	//cout<<"Set Title"<<endl;
	graph[0]->SetTitle("U Cal: Uset vs. Uout - ");
	//cout<<"Draw"<<endl;
	graph[0]->Sort();
	cout<<"Removed "<<graphCleaner(graph[0])<<" points"<<endl;
	graph[0]->GetXaxis()->SetTitle("U_dac [mV]");
	graph[0]->GetYaxis()->SetTitle("U_OUTPUT [mV]");
	graph[0]->Draw("A*L");

	cout<<"New graph 1"<<endl;
	tcan[i]->cd(2);
	graph[1]= new TGraph(string.Data(),"%*lg %lg %lg %*lg %*lg");
	
	
	graph[1]->SetTitle("U Cal: Uout vs. MonUreg - ");
	graph[1]->Sort();
	cout<<"Removed "<<graphCleaner(graph[1])<<" points"<<endl;
	graph[1]->GetXaxis()->SetTitle("U_OUTPUT [mV]");
	graph[1]->GetYaxis()->SetTitle("U_REGULATOR [mV]");
	graph[1]->Draw("A*L");

	tcan[i]->cd(3);
	graph[2]= new TGraph(string.Data(),"%*lg %lg %*lg %lg %*lg ");
	
	
	
	graph[2]->SetTitle("U Cal: Uout vs. MonUload - ");
	graph[2]->GetXaxis()->SetTitle("U_OUTPUT [mV]");
	graph[2]->GetYaxis()->SetTitle("U_LOAD [mV]");
	graph[2]->Sort();
	cout<<"Removed "<<graphCleaner(graph[2])<<" points"<<endl;
	graph[2]->Draw("A*L"); 

	string=path;
	string+=part1;
	string+=i;
	string+=ii;
	string+=part3;

	cout<<"Preparing Graphs:"<<string.Data()<<endl;

	tcan[i]->cd(4);
	graph[3]= new TGraph(string.Data(),"%*lg %lg %lg %*lg");
	
	
	graph[3]->SetTitle("I Cal: Iout vs. IoutMon");
	graph[3]->Sort();
	//cout<<"Removed "<<graphCleaner(graph[3])<<" points"<<endl;
	graph[3]->GetXaxis()->SetTitle("I_OUTPUT [mA]");
	graph[3]->GetYaxis()->SetTitle("U_MONITORING [mV]");
	
		if(i==13){ 	
				graph[3]->GetYaxis()->SetTitle("I_OUTPUT [muA]");	
				conv_to_mu_x(graph[3]);
				}
//************************************************
// For +-100mA call if condition must be removed 
//************************************************		

		if(i==15){ 	
				//graph[3]->GetYaxis()->SetTitle("I_OUTPUT [muA]");	
				//conv_to_dmu_x(graph[3]);
				}
	
	
	graph[3]->Draw("A*L");

	tcan[i]->cd(11);
	graph_help[0]= new TGraph(string.Data(),"%*lg %lg %*lg %lg  ");
	

	
	graph_help[0]->SetTitle("I Cal: Iout vs. UoutMon - horizontal line");
	
	graph_help[0]->Sort();
	graph_help[0]->Draw("A*L");
	
	tcan[i]->cd(12);
	graph_help[3]= new TGraph(string.Data(),"%*lg %lg %*lg %*lg %*lg %lg ");
	

	
	graph_help[3]->SetTitle("I Cal: Iout vs. U-SMU - horizontal line");
	//cout<<"Removed "<<graphCleaner(graph_help[3])<<" points"<<endl;
	graph_help[3]->Sort();
	graph_help[3]->Draw("A*L");

	tcan[i]->cd(14);
	graph_help[1]= new TGraph(string.Data(),"%*lg %lg %*lg %*lg %lg");

	graph_help[1]->SetTitle("I Cal: Iout vs. StatBit - should be high");
	graph_help[1]->Sort();
	graph_help[1]->Draw("A*L"); 
	
	string=path;
	string+=part1;
	string+=i;
	string+=ilim;
	string+=part3;

	cout<<"Preparing Graphs:"<<string.Data()<<endl;
	
	tcan[i]->cd(5);
	graph[4]= new TGraph(string.Data(),"%lg %lg %*lg %*lg %l*g");
	

	
	graph[4]->Sort();
	graph[4]->Sort();
	rem=graphCleaner(graph[4]);
	rem+=graphCleaner(graph[4]);
	cout<<"Removed "<<rem<<" points"<<endl;
	rem=0;
	graph[4]->SetTitle("I Cal: DAC LIMIT vs. I Measured");

	graph[4]->GetXaxis()->SetTitle("Limit DAC [mV]");
	graph[4]->GetYaxis()->SetTitle("Limit current [mA]");
	//cout<<"Removed "<<graphCleaner(graph[4])<<" points"<<endl;
	if(i==13){ 	graph[4]->GetYaxis()->SetTitle("Limit current [muA]");	
				conv_to_mu_y(graph[4]);
				}
	graph[4]->Draw("A*L"); 
	
	tcan[i]->cd(14);
	graph_help[2]= new TGraph(string.Data(),"%*lg %lg %*lg %*lg %lg");
	

	graph_help[2]->SetTitle("Limit Cal: I Limit out vs. StatBit - should be low");
	graph_help[2]->Sort();
	graph_help[2]->Draw("A*L"); 
	
	//cout<<" **************************************** "<<endl;
	//cout<<"Fit data"<<endl;

	for( int j=0; j<5;j++){
	

frac=.9;
 graph[j]->ComputeRange(comp_x_min,comp_y_min,comp_x_max,comp_y_max);
range=TMath::Abs(comp_y_max-comp_y_min);

upper_limit=0.95*range+comp_y_min;
lower_limit=0.05*range+comp_y_min;


//cout<<"Channel: "<<i<<" Graph: "<<j<<endl;
//cout<<"points: "<<graph[j]->GetN()<<endl;
//cout<<"Full y range: "<<range<<endl;
//cout<<"compute range "<<comp_x_min<<" "<<comp_x_max<<" "<<comp_y_min<<" "<<comp_y_max<<" "<<endl;
//cout<<"Y lower limit: "<<lower_limit<<endl;
//cout<<"Y upper limit: "<<upper_limit<<endl;

		num_pts=graph[j]->GetN();

		 int num_above=0;
lin[j]=new TF1("lin","[0]*x+[1]");
graph[j]->Fit(lin[j],"Q");
lin[j]->GetParameters(parameter2);


falling=true;
	if(parameter2[0]>0){ 
			falling=false;
			//cout<<"fit rising funct"<<endl;
			}
if(parameter2[0]<0) ;//	cout<<"fit falling funct"<<endl;
		
		
		 
		range_low=bisec(graph[j],lower_limit,falling); //fall[j]);
		range_high=bisec(graph[j],upper_limit,falling); //fall[j]);
		
//cout<<"x lower limit: "<<range_low<<endl;
//cout<<"y upper limit: "<<range_high<<endl;
		


//cout<<"Fit graph: "<<j<<endl;


		if(range_high<range_low){ 
		   place_holder=range_high;
		   range_high=range_low;
		   range_low=place_holder;
		}

//cout<<"Fit data between: "<<range_low<<" "<<range_high<<endl;

lin[j]->SetRange(range_low,range_high);

graph[j]->Fit(lin[j],"RQ");
	

		//cout<<"Get Parameter graph: "<<j<<endl;
		lin[j]->GetParameters(parameter2);
		param[i][j][0]=parameter2[0];
		param[i][j][1]=parameter2[1];
		//cout<<"Offset: "<<parameter2[0]<<"Gain: "<<parameter2[1]<<endl;
	
	//cout<<"producing residuals"<<endl;
	string="Residuum ";
	string+=j;
	res_graph[j]=new TGraph();
	res_graph[j]->SetName(string.Data());
			
			int counter=0;
			
			for(int k=1;k<(graph[j]->GetN()-1);k++){

						graph[j]->GetPoint(k, x, y);
					//	cout<<x<<" : "<<y<<endl;
						resi=y-lin[j]->Eval(x);
						res_graph[j]->SetPoint(k-1,x,resi);
						
						// removing data points 
						
						rel_resi = resi/y;
						if (rel_resi<0){
							rel_resi*=(-1);
						}
						if (rel_resi > 0.02) {			// 
							//graph[j]->RemovePoint(k);
							//res_graph[j]->RemovePoint(k-1);
							counter++;
						} 

						
						
					}
					if (counter > 0){
						std::cout<<"manually removed: " << counter << " points " << std::endl; 
					}

//cout<<"Draw residuum of graph "<<j<<" into canvas "<<j+6<<endl;
tcan[i]->cd(j+6);
res_graph[j]->GetXaxis()->SetRangeUser(range_low,range_high);
graph[j]->Draw("A*L");
res_graph[j]->Draw("A*");


if (j==3){
	lin[j]=new TF1("lin","[0]*x+[1]");
	graph_help[j]->Fit(lin[j],"Q");
	lin[j]->GetParameters(parameter2);
	// std::cout << "parameter2[0] = " << parameter2[0] << "    parameter2[1] = " << parameter2[1] << std::endl;
	
	TString p0="";
	TString p1="";
	
	tcan[i]->cd(13);
	p0 += parameter2[0];
	p1 += parameter2[1];	
	TLatex text;
	//p0.Form("%4.2f", p0);
	//p1.Form("%4.8f",p1);
	//text.DrawLatex(0.2, 0.6, p0.Data());
	//text.DrawLatex(0.2, 0.4, p1.Data());
}
	}// channel fit loop

pathname = Form("/home/silab44/pxd_teststand_software_frederik/data/example/plots/Channel%d.pdf", i);

cout << endl << "The path is " << pathname << endl;
tcan[i]->SaveAs(pathname);	
	
	

	

cout<<" **************************************** "<<endl;
cout<<endl;	
	


} // if
else{ // data file not availiable --> no card installed --> set to default
	for( int j=0; j<5;j++){

		param[i][j][0]=1.0;
		param[i][j][1]=0.0;
		}
}


} // end of channel loop


cout<<"Open callibration File"<<endl;

TString calli;
ofstream myfile;

//TString filename="C:/root/Callibration/constants_node_";
TString filename="/home/silab44/pxd_teststand_software_frederik/data/example/constants_node_";

filename+=node;
filename+=".ini";

 myfile.open (filename.Data());

TString glob="[GLOBAL]";
myfile << glob.Data()<<endl;

glob="NODE=";
glob+=node;
myfile << glob.Data()<<endl;

glob="NUM_CHANNELS=";
glob+=num_files;
myfile << glob.Data()<<endl;
myfile << "" <<endl;

TString channel_tag; //="[CHANNEL];

for( int i=0; i<num_files;i++){
	calli="";
	channel_tag="[";
	channel_tag+=i;
	channel_tag+="]";
	myfile << channel_tag.Data()<<endl;
	
	for( int j=0; j<5;j++){
            	   
	    calli=tags[j];
            calli+=gain;
            calli+="=";	
            calli+=TMath::Nint(param[i][j][0]*10000);
	    myfile << calli.Data()<<endl;

            	    
	    calli=tags[j];
	    calli+=offset;
        calli+="=";	
		//debug
		// if(i == 16) std::cout << param[i][j][1] << " " << TMath::Nint(1.0*param[i][j][1]*100) << std::endl;
        calli+=TMath::Nint(1.0*param[i][j][1]*100); // mult by 100 to fit into firmware
	    myfile << calli.Data()<<endl;
	
		//cout<<tags[j].Data()<<" Offset: "<<param[i][j][1]<<" Gain: "<<param[i][j][0]<<endl;
	
  
			} // for all measurements




	} // for all channels

  myfile.close();

 int parsed=0;
 
for(int i=0;i<num_files;i++){ parsed=parsed+data_available[i];}

cout<<"Found "<<parsed<<" callibration files"<<endl;


}
