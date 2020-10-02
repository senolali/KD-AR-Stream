%%%%%%%%%%%%%KD-AR Stream Matlab Code%%%%%%%%%%%%%%%%%%%%%%%%
% This Code is imlementation of article which proposed Ali �enol and Hacer
% Karacan. Please if you use the code cite the papaer below:
% �enol, A , Karacan, H . (2019). Kd-tree and adaptive radius (KD-AR Stream) based real-time data stream clustering . 
%Journal of the Faculty of Engineering and Architecture of Gazi University , 35 (1) , 337-354 . DOI: 10.17341/gazimmfd.467226
%
%NOte 1: Because of using time based summarization, buffered data in the
%setted time could be different in different computers because of different
%performance of each computer. 
%
%Note 2: This code is optimised after publication of paper. So some parameter
%settings may be different from that used in the article.
% 
%Note: 3 Occupancy dataset is used to test the poposed apprach with the
%datat that has timestamp. For this reason we firstley manipulate the
%timestamp of each data.
%
%Assist. Prof. Dr. Ali �enol 
%Department of Computer Engineering, Faculty of Engineering and Natural
%Sciences, Gaziantep Islam Science and Technology University,
%ali.senol@gibtu.edu.tr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear all;
addpath data

veriseti=2;% Datasets to be selected:  1=idealVeri, 2=ExclaStar, 3=Fisheriris, 4=KDD,  8=MrData,9=Breast  11=Occupancy 
plotGraph=1; %If plot will be drawn
if veriseti==1 %Selected dataset IdealData Produced Sysntetic Dataset which produced bu us 
    %%%%Dataset Initializing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    load('IdealData.mat');
    load('IdealDataLabels.mat');
    DatasetName='idealVeri_';
    VeriMiktari=211;              %Size of Dataset
    d=2;                           %# of features(d=dimensions)
    data=AllData(1:VeriMiktari,1:d);%Selected Part of Dataset
    class=ACTUAL(1:VeriMiktari,1); %class labels of dataset
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%Parameter Setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=3;%Minimum number of data required to create a cluster
    t=0.05; %Data lifetime threshold value
    TN=30;%If data size is too big, amount of data taken as summary
    r=4; %Initial cluster radius / shell radius
    r_treshold=3; %Radius increment threshold
    r_max=9; %Maximum value for radius
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    %%%%%Programming variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S=[]; %Cluster table
    buffer=zeros(1,d+4); %empty data matrix - Processed data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
if veriseti==2  %%%ExclaStar Dataset which was produced by DPStream
    dataset=load('DatasetNew.txt');
    DatasetName='ExclaStar_';
    VeriMiktari=755;              
    d=2;                           
    data=dataset(1:VeriMiktari,2:d+1);
    class=dataset(1:VeriMiktari,4);

    % %User Defined Variables%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=20;
    t=setTime('00-01-0000 00:00:03 000');
    TN=65;
    r=4.5;
    r_treshold=2.75;
    r_max=11.25;
    
    S=[];
    buffer=zeros(1,d+4);
end
if veriseti==3 %%%Fisher Iris Dataset
    
    load 'fisheriris';
    DatasetName='Fisheriris_';
    VeriMiktari=150;              
    d=4;                           
    data=meas(1:VeriMiktari,1:d);
    class=grp2idx(species);
    class=class(1:VeriMiktari,1);
 
  % %User Defined Variables%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=7;
    t=0.03;
    TN=80;
    r=1.05;
    r_treshold=0.45;
    r_max=1.8;
    
    buffer=zeros(1,d+4);
    S=[];
end
if veriseti==4 %%KDD Dataset 
    load('KDD_Sample_Dataset.mat'); 
    DatasetName='KDD_';
    VeriMiktari=50000;
    d=38;                           
    data=dataset(1:VeriMiktari,1:d);
    class=labels(1:VeriMiktari,:);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%Class Relabelling %%%%%%%%%%%%%%
    class(class(:,1)==1,1)=111;
    class(class(:,1)==2,1)=112;
    class(class(:,1)==3,1)=113;
    class(class(:,1)==4,1)=114;
    class(class(:,1)==5,1)=115;
    class(class(:,1)==6,1)=116;
    class(class(:,1)==7,1)=117;
    class(class(:,1)==8,1)=118;
    class(class(:,1)==9,1)=119;
    class(class(:,1)==10,1)=120;
    class(class(:,1)==11,1)=121;
    class(class(:,1)==12,1)=122;
    class(class(:,1)==13,1)=123;
    class(class(:,1)==14,1)=124;
    class(class(:,1)==15,1)=125;
    class(class(:,1)==16,1)=126;
    class(class(:,1)==17,1)=127;
    class(class(:,1)==18,1)=128;
    class(class(:,1)==19,1)=129;
    class(class(:,1)==20,1)=130;
    class(class(:,1)==21,1)=131;
    
    
    class1=class;
    class(class(:,1)==122,1)=1;
    class(class(:,1)==129,1)=2;
    class(class(:,1)==131,1)=3;
    class(class(:,1)==116,1)=4;
    class(class(:,1)==121,1)=5;
    
    class1(class1(:,1)==122,1)=1;
    class1(class1(:,1)==129,1)=2;
    class1(class1(:,1)==131,1)=3;
    class1(class1(:,1)==116,1)=4;
    class1(class1(:,1)==121,1)=5;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    N=90;
    TN=160;
    t=20; % 20s selected to guarantee getting nearly the same results as the paper 
    r=1.5;
    r_treshold=3;
    r_max=4.5;
    
    buffer=zeros(1,d+4);
    S=[];
end
if veriseti==8 % MrData which is used in DPStream
    dataset=load('MrData.txt');
    DatasetName='MrData_';
    VeriMiktari=42470;             
    d=2;                           
    data=dataset(1:VeriMiktari,2:d+1);
    class=dataset(1:VeriMiktari,4);
    
    %class relabelling%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    class(class(:,1)==1,1)=1;
    class(class(:,1)==4,1)=40;
    class(class(:,1)==2,1)=20;
    class(class(:,1)==3,1)=30;
    class(class(:,1)==40,1)=2;
    class(class(:,1)==20,1)=3;
    class(class(:,1)==30,1)=4;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
   % %User Defined Variables%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=25;
    t=0.1;
    TN=200;
    r=15;
    r_treshold=2.5;
    r_max=26;
    
    buffer=zeros(1,d+4);
    S=[];
end
if veriseti==9 %Breast Cancer Dataset
    dataset=importdata('Breast_Cancer.txt');
    DatasetName='BreastCancer_';
    VeriMiktari=699;              
    d=9;                           
    data=dataset(1:VeriMiktari,2:d+1);
    class=dataset(1:VeriMiktari,11);
    class(class(:,1)==2,1)=1;
    class(class(:,1)==4,1)=2;
    

   % %User Defined Variables%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=3;
    t=0.3;
    TN=200;
    r=8.5;
    r_treshold=2.5;
    r_max=11.5;%
   
    buffer=zeros(1,d+4);
    S=[];
end
if veriseti==11 %Occupancy Dataset
    load('Occupancy_Data.mat');
    DatasetName='Occupancy_';
    VeriMiktari=8143;              
    d=6;                          
    data=Occupancy_Data(1:VeriMiktari,2:d+1);
    class=Occupancy_Data(1:VeriMiktari,8);
       
    simdi=now;
    for abc=1:VeriMiktari
       if mod(rand(1,1)*10,2)==0
        if abc==1
            Time_Mat(abc,1)=simdi+0.000001*rand(1,1);
        else
            Time_Mat(abc,1)=Time_Mat(abc-1,1)+0.000001*rand(1,1);
        end
       else
            if abc==1
            Time_Mat(abc,1)=simdi+0.00001*rand(1,1);
        else
            Time_Mat(abc,1)=Time_Mat(abc-1,1)+0.00001*rand(1,1);
        end
       end
    end
    Time_Mat2=datestr(Time_Mat,'dd-mm-yyyy HH:MM:SS FFF')
     
    class(class(:,1)==1,1)=2;
    class(class(:,1)==0,1)=1;

   % %User Defined Variables%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    N=20;
    t=setTime('00-01-0000 00:00:05 000');
    TN=350;
    r=314;
    r_treshold=60;
    r_max=395;
    
    buffer=zeros(1,d+4);
    S=[];
end    

[ii,jj]=find(isnan(data)| isinf(data));
data(ii,jj)=0;
i=1;
figureciz=0;
flag=true;
display('Started...');
Pred=[];

% %set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.25, 0.5, 0.75]);
x0=300;
y0=100;
width=500;
height=450;
set(gcf,'units','points','position',[x0,y0,width,height]);

bufSize=[];
Pred2=[];
Silinen=[];

indeksler = arrayfun(@(n) 1:n, VeriMiktari, 'UniformOutput', false);
indeksler=indeksler{1,1}';
%data[indexNo d1 d2 d3 classLabel PredictedClus(Model ekliyor)]
data=[indeksler data class];
doDimensionReduction=0;
dilk=d;
SD=zeros(1,d);
Z=[];
tic;
while flag==true %Cycle of the algorithm
        
    buffer=addNode2(data(i,1:d+1),buffer,d,i); %Get new data
    [S,buffer,Pred,Silinen]=AgeingAll(S,buffer,d,t,Pred,N,Silinen,TN); %Age all data and delete data expired
    [S,buffer]=NewClusterAppear(S,buffer,d,r,N); %Search for new clusters
    [S,buffer]=checkOverlapClustering(S,buffer,N,d,i); %Control if there are any clusters overlapped
    [S,buffer]=checkSplit(S,buffer,N,r,d,i); %Control if there any cluster to be split
    [S,buffer]=findAndAddClosestCluster(S, buffer,N,r_max,r_treshold,d); %Find closest cluster to the new arrived data. If there is any cluster sufficiently close assign
    [S,buffer]=FlagActiveCluster(S,buffer,N,d); %Active clusters which have sufficient data, inactive the others
    [S,buffer]=updateCenters(S,buffer,d,N,r,Pred,TN);%Update center of all clusters
    [S,buffer]=updateRadius(S,buffer,r_max,d,N,r);%Update core and shell radii of all clusters
    
    if size(Silinen,1)>0 %%Deleted expired data and add those data to deleted index
        Pred2=[Pred2;[Silinen(:,1) Silinen(:,end-2)]]; 
        Silinen=[];
    end
    if plotGraph==1 %If ploth will be done  do it
        if d == 3 %For 3D diensions
                myplot(S,buffer,figureciz,d,r_max,veriseti)
        end
        if d == 2 %%For 2D dimensions
                  myplot(S,buffer,figureciz,d,r_max,veriseti)
        end
    end
    if i>VeriMiktari-1
        flag=false;
        
    else
       if veriseti==11 %% if dataset is Occupancy
            next=Time_Mat(i+1,1);
            current=now;
            while 30*current<next
               pause(0.001); 
               current=now;
            end
            
       end   
    end
    i=i+1;
    if mod(i,1000)==0
        fprintf('i=%d \n',i);        
    end
    bufSize(1,i)=size(buffer,1);
    elapsedTime(1,i)=toc;
    elapsedTime(2,i)=size(buffer,1);
    elapsedTime(3,i)=elapsedTime(1,i)-elapsedTime(1,i-1);
end
tok=toc;
fprintf('Elapsed Time: %f\n',tok);

string1='TestResults/';
string4=num2str(VeriMiktari);
adres = strcat(string1,DatasetName);
adres=strcat(adres,string4);

data=[data [Pred2(:,2);buffer(:,d+2)]];
CompareMatrix=data(:,d+2:d+3);

for i=1:size(CompareMatrix,1)
PREDICTED=CompareMatrix(1:i,1);
ACTUAL=CompareMatrix(1:i,2);
%        i 
    %%%%%%%%%%%%%Purity%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    p=Purity(ACTUAL,PREDICTED);
    PurityMatric(1,i)=p;
%     PurityMean(1,i)=mean(PurityMatric(1,1:i));
    OverallPurity=mean(PurityMatric(1,1:i));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    EVAL = Evaluate(ACTUAL,PREDICTED);
    %%%%%%%%%%%%%Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Accurecy(1,i)=EVAL(1,1);
%     AccMeans(1,i)=mean(Accurecy(1,1:i));
    OverallAcc=mean(Accurecy(1,1:i));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%F-Score%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~isnan(EVAL(1,6))
        Precision(1,i)=EVAL(1,4);
        Recall(1,i)=EVAL(1,5);
        FScores(1,i)=EVAL(1,6);
    else
        Precision(1,i)=1;
        Recall(1,i)=1;
        FScores(1,i)=1;
    end
%     FscoreMean(1,i)=mean(FScores(1,1:i));
    OverallPrecision=mean(Precision(1,1:i));
    OverallRecall=mean(Recall(1,1:i));
    OverallFScore=mean(FScores(1,1:i));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%    %%%%%%%%%%%%%%Adjusted Rand Index %%%%%%%%%%%%%%%%%%%%%%%%%%%
%    if i>1
       AR=EVAL(1,6);
       if ~isnan(AR)
         RandMatrix(1,i) = AR;
       else
          RandMatrix(1,i)=1;
       end
%        RandMean(1,i)=mean(RandMatrix(1,1:i));
       OverallRand=mean(RandMatrix(1,1:i));
%    else
%        RandMatrix(1,i)=1;
% %        RandMean(1,i)=mean(RandMatrix(1,1:i));
%        OverallRand=mean(RandMatrix(1,1:i));
%    end
%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     
%    %%%%%%%%%NMI%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     nmiscore = nmi(ACTUAL, PREDICTED);
%     NMIMatrix(1,i) = nmiscore;
% %     NMIMean(1,i)=mean(NMIMatrix(1,1:i));
%     OverallNMI=mean(NMIMatrix(1,1:i));
%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    
   %%%%%%%%%%%%%%Average Elapsed Time Per Data%%%%%%%%%%%%%%%%%%%%
    if i~=1     
         AvgElapsedTime(1,i)=elapsedTime(3,i) /elapsedTime(2,i); 
    else
        AvgElapsedTime(1,i)=0;
    end
    OverallAvgElapsedTime=mean(AvgElapsedTime);
    % AvgElapsedTime(1,i)=mean(AvgElapsedTime1(1,1:i));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if mod(i,1000)==0
       fprintf('i=%d \n',i);        
    end
end
OverallAcc
%%%%%%%%%%%%%%%%Silheoutte%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Gs=  silhouette(data(:,2:d+1),data(:,d+3),'Euclidean');
GMatrix = Gs;
OverallG=mean(Gs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
islem='_CompareMatrix.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,CompareMatrix);

islem='_Purity.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,PurityMatric);

islem='_RandIndex.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,RandMatrix);
%   
% islem='_NMI.txt';
% PurDosya=strcat(adres,islem);
% dlmwrite(Dosya,NMIMatrix);

islem='_Accurecy.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,Accurecy);

islem='_Precision.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,Precision);

islem='_Recall.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,Recall);

islem='_FScore.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,FScores);

islem='_Silheoutte.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,GMatrix);

adres=strcat(adres,num2str(d));
islem='_ElapsedTime.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,elapsedTime);

islem='_ElapsedTimePerData.txt';
Dosya=strcat(adres,islem);
dlmwrite(Dosya,AvgElapsedTime);



figure (3)
plot(PurityMatric(1, 1:i),'LineWidth',2)
ylim([0 1])
title('Purity Test')
xlabel('Veri Miktar�')
ylabel('Purity')
set(gcf,'units','points','position',[x0,y0,width,height]);

figure (4)
plot(Accurecy(1, 1:i),'LineWidth',2);
ylim([0 1])
title('Accurecy Test')
xlabel('Veri Miktar�')
ylabel('Accurecy')
set(gcf,'units','points','position',[x0,y0,width,height]);
% 
 figure (5)
plot(FScores(1, 1:i),'LineWidth',2);
ylim([0 1])
title('F-Score Test')
xlabel('Veri Miktar�')
ylabel('F-Score')
set(gcf,'units','points','position',[x0,y0,width,height]);

 figure (6)
plot(Precision(1, 1:i),'LineWidth',2);
ylim([0 1])
title('Precision Test')
xlabel('Veri Miktar�')
ylabel('Precision')
set(gcf,'units','points','position',[x0,y0,width,height]);

 figure (7)
plot(Recall(1, 1:i),'LineWidth',2);
ylim([0 1])
title('Recall Test')
xlabel('Veri Miktar�')
ylabel('Recall')
set(gcf,'units','points','position',[x0,y0,width,height]);

% 
figure (8)
plot(RandMatrix(1, 1:i),'LineWidth',2);
ylim([0 1])
title('Adjustes Rand Index')
xlabel('Veri Miktar')
ylabel('Rand Index')
set(gcf,'units','points','position',[x0,y0,width,height]);
% 
% figure (9)
% plot(NMIMatrix(1, 1:i),'LineWidth',2);
% ylim([0 1])
% title('NMI')
% xlabel('Veri Miktar')
% ylabel('NMI')
% set(gcf,'units','points','position',[x0,y0,width,height]);
% 
figure (10)
plot(GMatrix,'LineWidth',2);
% ylim([0 1])
title('Global Silheoutte')
xlabel('Veri Miktar�')
ylabel('Silheoutte Index')
set(gcf,'units','points','position',[x0,y0,width,height]);
% 
figure (11)
plot(elapsedTime(1, 1:i),'LineWidth',2);
title('Time Complexity')
xlabel('Veri Miktar�')
ylabel('Elaped Time(sn)')
set(gcf,'units','points','position',[x0,y0,width,height]);

figure (12)
plot(AvgElapsedTime(1, 1:i),'LineWidth',2);
ylim([0 1])
title('Run Time Per Data')
xlabel('Veri Miktar�')
ylabel('Average Time(sn)')
set(gcf,'units','points','position',[x0,y0,width,height]);


figure (13)
plot(bufSize(1, 1:i),'LineWidth',2);
hold on
ylim([0 500])
title('# of Data Processed in each Step')
ylabel('Veri Say�s�')
xlabel('Veri Miktar�')
set(gcf,'units','points','position',[x0,y0,width,height]);

fprintf('Run Time Per Data=%d\n',OverallAvgElapsedTime);
%  fprintf('AvgNMI=%d\n',OverallNMI);
 
 fprintf('AvgPrecision=%d\n',OverallPrecision);
 fprintf('AvgRecall=%d\n',OverallRecall);
 fprintf('AvgFScore=%d\n',OverallFScore);
 fprintf('AvgARI=%d\n',OverallRand);
 fprintf('AvgSilhoutte=%d\n',OverallG);
 fprintf('AvgPurity=%d\n',OverallPurity);
 fprintf('AvgAccuracy=%d\n',OverallAcc);

      