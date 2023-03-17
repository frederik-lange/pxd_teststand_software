
void conv_to_mu_y(TGraph *tg){
int n_pts=tg->GetN();
double x,y;
int i=0;
for(i=0;i<n_pts;i++){
	tg->GetPoint(i,x,y);
	tg->SetPoint(i,x,1000*y);
	}	
	return  ;
}

void conv_to_mu_x(TGraph *tg){
int n_pts=tg->GetN();
double x,y;
int i=0;
for(i=0;i<n_pts;i++){
	tg->GetPoint(i,x,y);
	tg->SetPoint(i,1000*x,y);
	}	
	return  ;
}


void conv_to_dmu_x(TGraph *tg){
int n_pts=tg->GetN();
double x,y;
int i=0;
for(i=0;i<n_pts;i++){
	tg->GetPoint(i,x,y);
	tg->SetPoint(i,10000*x,y);
	}	
	return  ;
}




int graphCleaner(TGraph *graph){
	
double quant[2];
double prob[2] ;
prob[0]=0.3;
prob[1]=0.7;
double robustRmsY,robustRmsX;
double X_Win[15],Y_Win[15],X_WinSort[10],Y_WinSort[10];
int index[15];
int numPoints=graph->GetN();
double x,y,Xmed,Ymed,Yrms,Xrms;
int remPoints=0;
// der zu testende punkt sollte nicht mit in der statistik sein... 
// problem am oberen rand... 
for(int k=0;k<(numPoints);k++){

	graph->GetPoint(k, x, y);
	//	cout<<x<<" : "<<y<<endl;

	for( int i=0;i<15;i++){
	   X_Win[i]=Y_Win[i]=0;
	   
		if( (numPoints-k) > 15) graph->GetPoint(k+i, X_Win[i], Y_Win[i]);
	
		if( (numPoints-k)<=15) graph->GetPoint(numPoints-15+i,X_Win[i], Y_Win[i]); 
	
		}
	
	TMath::Sort(15,Y_Win,index,kTRUE);
	for(int j=2;j<12;j++){Y_WinSort[j-2]=Y_Win[index[j]];}
	
	TMath::Sort(15,X_Win,index,kFALSE);	
	for(int j=2;j<12;j++){X_WinSort[j-2]=X_Win[index[j]];}
	
	Xmed=TMath::Median(15,X_Win);
	Xrms=TMath::RMS(10,X_WinSort);
	Ymed=TMath::Median(15,Y_Win);
	Yrms=TMath::RMS(10,Y_WinSort);
	
	
	//TMath::Quantiles(15,2,Y_Win,quant,prob,kFALSE);
	//robustRmsY=TMath::Abs(3*(quant[1] - quant[0]));
	
	//TMath::Quantiles(15,2,X_Win,quant,prob,kFALSE);
	//robustRmsX=TMath::Abs(3*(quant[1] - quant[0]));
	
	graph->GetPoint(k, x, y);
//cout<<"y: "<<y<<" y-Ymed "<<y-Ymed<<" yrms: "<<Yrms<<" robustRms " <<robustRmsY<<endl;
//cout<<x<<" : "<<y<<" y -Ymed "<<y-Ymed<<" yrms: "<<Yrms<<endl;
//cout<<x<<" y -Ymed "<<x-Xmed<<" yrms: "<<Xrms<<endl;
	if ( (TMath::Abs((x-Xmed)*1.0/Xrms) > 5) || (TMath::Abs((y-Ymed)*1.0/Yrms) > 5) ){
		
		graph->RemovePoint(k);
		
		 cout<<"Removal: y -Ymed "<<y-Ymed<<" yrms: "<<Yrms<<endl;
		
		remPoints++;	}
	
	
	
	}
	
	
	return remPoints;
					
					
	}



