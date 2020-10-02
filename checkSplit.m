function [S,datas]=checkSplit(S,datas,N,r,d,iii)

    for k=1:findNumOfAllClusters(S)
      if S(k,d+2)>=2*N  && S(k,d+4)>2*r && S(k,d+3)==1 && S(k,d+8)>=r
        data=datas(datas(:,d+2)==k,:);%Hi�bir k�meye ait olmayan verileri al
        tree = KDTreeSearcher(data(:,2:d+1));%Bu verilerin/sat�rlar�n niteliklerini al ve a�aca yerle�tir
        
        for i=1:size(data,1)%Ka� tane veri varsa her biri i�in kontrol edilecek      
             query = data(i,2:d+1);%Bu verilerin her biri i�in sorgu olu�turulacah
             idx = rangesearch(tree, query, r);%Yar��ap verisine g�re range search yap
             idxs = idx{1};%r yar��ap i�erisinde bulunan verilerin indexlerini al
      
             
             if size(idxs,2)>=N %E�er r i�erisindeki veri say�s� k�me olu�turacak say�da ise  gir
                 A=data(idxs,1);%%data matrisinde bu verilerin id(ilk s�tun) de�erlerini al
                 B=sortrows(A);%S�rala                       
                 datax=datas( ismember(datas(:,1) ,B(:,1),'rows' ),: );%%AA K�mesi
                 datay=data( ~ismember(data(:,1) ,B(:,1),'rows' ),: );%BB k�mesi
                 
                 for l=1:d
                     AA(1,l)=mean(datax(:,l+1));
                     BB(1,l)=mean(datay(:,l+1));
                 end
                 
                 
                 dis=0;
                 rX=0;
                 
                 for m=1:size(datax,1)
                     dis=findDistance(datax(m,2:d+1),AA(1,1:d),d);
                     if dis>rX
                         rX=dis;
                     end                     
                 end
                 dis=0;
                 rY=0;
                 for m=1:size(datay,1)
                     dis=findDistance(datay(m,2:d+1),BB(1,1:d),d);
                     if dis>rY
                         rY=dis;
                     end                     
                 end 
                 
                 if size(datax,1)>=N && size(datay,1)>=N
                    dist=findDistance(AA,BB,d);
                    if dist>((rX+rY)+r/2)
                        newClusterNo=findNumOfAllClusters(S)+1;

                         datas( ismember(datas(:,1) ,B(:,1),'rows' ),d+2 )=newClusterNo;
                         datas( ismember(datas(:,1) ,B(:,1),'rows' ),d+4 )=1;
                         [S]=addNewCluster(S,newClusterNo,datax,d,r);
                         fprintf('%d nolu k�me b�l�nd�. %d nolu k�me olu�turuldu. i= %d\n',k,newClusterNo,iii);
                         S(k,d+9)=1;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                         S(newClusterNo,d+9)=1;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                         return;
                    end
                     
                 end
                 
                
             end
        end
      end
    end
end