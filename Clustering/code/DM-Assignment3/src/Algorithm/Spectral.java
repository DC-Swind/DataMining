package Algorithm;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import Jama.Matrix;

public class Spectral {
	public static int clusterN = Setting.clusterN;
	public static int featureN = Setting.featureN;
	public static int dataN = Setting.dataN;
	public static double means[][] = new double[clusterN][featureN];
	public static double newmeans[][] = new double[clusterN][featureN];
	public static int newmeansN[] = new int[clusterN];
	public static double data[][] = new double[dataN][featureN];
	public static int datatag[][] = new int[dataN][2];
	public static double ldata[][] = new double[dataN][clusterN];
	
	public static int edge = 3;
	
	public static Map<Integer,Map<Integer,Double>> Knear = new HashMap<Integer,Map<Integer,Double>>();
	public static Matrix W = new Matrix(dataN,dataN);
	
	public static double distance(double[] a,double[] b){
		double dis = 0;
		for(int i=0; i<featureN; i++){
			dis += (a[i]-b[i]) * (a[i]-b[i]);
		}
		return dis;
	}
	
	
	
	public static void construct(int n){
		System.out.print("construct graph...	");
		for(int i=0; i<dataN; i++){
			Map<Integer,Double> map = new HashMap<Integer,Double>();
			Knear.put(i, map);
		}
		
		for(int i=0; i<dataN; i++){
			double dis[] = new double[dataN];
			
			for(int j=0; j<dataN; j++) dis[j] = distance(data[i],data[j]);
			
			firstnitem(dis,i,n);
		}
		
		for(int i=0; i<dataN; i++){
			for(int j=0; j<dataN; j++)
				if (Knear.get(i).get(j) == null){
					
				}else{
					W.set(i,j,1.0);
				}
		}
		System.out.println("[done]");
	}
	
	private static void firstnitem(double[] dis, int index, int n) {
		int nn = n;
		while(nn > 0){
			double max = 99999999;
			double min = max;
			int mini = -1;
			for(int i=0; i<dataN; i++) if (dis[i] < min){
				min = dis[i];
				mini = i;
			}
			
			dis[mini] = max;
			Knear.get(index).put(mini, 1.0);
			Knear.get(mini).put(index, 1.0);
			nn--;
		}
		
	}



	public static void spectral(int n){
		construct(n);
		System.out.print("calculate low dimension represent..");
		Matrix D = new Matrix(dataN,dataN);
		for(int i=0; i<dataN; i++){
			double sum = 0;
			for(int j=0; j<dataN; j++) sum += W.get(i, j);
			D.set(i, i, sum);
		}
		Matrix L = D.minus(W);
		Matrix eigvector = L.eig().getV();
		Matrix eigvalue = L.eig().getD();
		
		System.out.print(".	");
		
		
		for(int i=0; i<clusterN; i++){
			double min = 99999999;
			int minj = -1;
			for(int j=0; j<dataN; j++) if (eigvalue.get(j,j) < min){
				min = eigvalue.get(j, j);
				minj = j;
			}
			eigvalue.set(minj, minj, 9999999);
			
			for(int j=0; j<dataN; j++){
				ldata[j][i] = eigvector.getArray()[j][minj];
			}
		}
		
		System.out.println("[done]");
	}
	
	public static void main(String[] args) throws IOException{
		Setting.fileInput(data, datatag);
		spectral(edge);
		Kmeans.featureN = clusterN;
		Kmeans.data = ldata;
		Kmeans.datatag = datatag;
		Kmeans.kmeans();
		Validation.init(Kmeans.datatag);
		System.out.println("purity is: " + Validation.purity());
		System.out.println("gini is:   " + Validation.gini());
		
		
	}
}
