
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
	prob=[1]=0.7;
	double robustRmsY,robustRmsX;
	double X_Win[15],Y_Win[15],X_WinSort[10],Y_WinSort[10];
	int index[15];
	int numPoints=graph->GetN();
	double x,y,Xmed,Ymed,Yrms,Xrms;
	int remPoints=0;
	// der zu testende punkt sollte nicht mit in der statistik sein... 
	// problem am oberen rand...
	
	// for loop iterating over all points of the graph 
	for(int k=0;k<(numPoints);k++){
		
		// get x and y coordinate of point k
		graph->GetPoint(k, x, y);
		//	cout<<x<<" : "<<y<<endl;

		// look at 15 values "in the vicinity" of point k
		for( int i=0;i<15;i++){
		   X_Win[i]=Y_Win[i]=0;
		   
		   	// for most points this will be true: is current point more than 15 points away from last point
			if( (numPoints-k) > 15) graph->GetPoint(k+i, X_Win[i], Y_Win[i]);
		
			// for points close to last point, take point numpoint-15+i
			if( (numPoints-k)<=15) graph->GetPoint(numPoints-15+i,X_Win[i], Y_Win[i]); 
		
			}
		
		// sort y coordinates by size (?)
		TMath::Sort(15,Y_Win,index,kTRUE);
		// take only "middle 12" of the previously selected 15 points
		for(int j=2;j<12;j++){Y_WinSort[j-2]=Y_Win[index[j]];}
		
		// same thing for x
		TMath::Sort(15,X_Win,index,kFALSE);	
		for(int j=2;j<12;j++){X_WinSort[j-2]=X_Win[index[j]];}
		
		// calculate median of 15 points and RMS of "middle 12 points" of the 15 points
		Xmed=TMath::Median(15,X_Win);
		Xrms=TMath::RMS(10,X_WinSort);
		Ymed=TMath::Median(15,Y_Win);
		Yrms=TMath::RMS(10,Y_WinSort);
		
		
		//TMath::Quantiles(15,2,Y_Win,quant,prob,kFALSE);
		//robustRmsY=TMath::Abs(3*(quant[1] - quant[0]));
		
		//TMath::Quantiles(15,2,X_Win,quant,prob,kFALSE);
		//robustRmsX=TMath::Abs(3*(quant[1] - quant[0]));
		
		// does nothing new
		graph->GetPoint(k, x, y);
	//cout<<"y: "<<y<<" y-Ymed "<<y-Ymed<<" yrms: "<<Yrms<<" robustRms " <<robustRmsY<<endl;
	//cout<<x<<" : "<<y<<" y -Ymed "<<y-Ymed<<" yrms: "<<Yrms<<endl;
	//cout<<x<<" y -Ymed "<<x-Xmed<<" yrms: "<<Xrms<<endl;
	
		// check if x coordinate is less than RMS away from x median of next 15 points, same for y
		if ( (TMath::Abs((x-Xmed)*1.0/Xrms) > 5) || (TMath::Abs((y-Ymed)*1.0/Yrms) > 5) ){
			
			// if that is the case: throw away point
			graph->RemovePoint(k);
			
			 cout<<"Removal: y -Ymed "<<y-Ymed<<" yrms: "<<Yrms<<endl;
			
			// count number of removed points
			remPoints++;	}
		
		
		
		}
		
	
	return remPoints;
					
					
}


/* python example for "sliding window around point"

for i, coord_x in enumerate(x):
    window_x_left = list(x[max(0,i-7):i])
    window_x_right = list(x[i+1:min(i+9,len(x))])
    print(f"point {i}: window {window_x_left+window_x_right}")
    if i == 10:
        break

*/

