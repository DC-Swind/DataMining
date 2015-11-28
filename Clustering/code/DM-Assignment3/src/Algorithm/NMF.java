package Algorithm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Time;
import java.text.DecimalFormat;

import Jama.Matrix;

public class NMF {
	public static int dataN = Setting.dataN;
	public static int featureN = Setting.featureN;
	public static int clusterN = Setting.clusterN;
	public static double data[][] = new double[dataN][featureN];
	public static int datatag[][] = new int[dataN][2];
	public static int realtag[][] = new int[dataN][2];
	public static Matrix X;
	public static Matrix U;
	public static Matrix V;
	
	public static void test(){
		Matrix X = Matrix.random(3, 2);
		
		System.out.println(X.getRowDimension());
		X.print(3, 3);
	}
	
	public static void output(String filename) throws IOException{
		DecimalFormat df=new DecimalFormat("0.000");
		System.out.print("print answer...	");
		File file = new File(filename);
		if (!file.exists()) file.createNewFile();
		BufferedWriter f = new BufferedWriter(new FileWriter(file));
		Matrix error = U.times(V.transpose());
		for(int i=0; i<error.getColumnDimension(); i++){
			for(int j=0; j<error.getRowDimension(); j++)
				f.write(df.format(error.get(j, i))+" ");
			f.newLine();
		}
		
		f.close();
		System.out.println("[done]");
	}
	
	public static void nmf(int iterN){
		X = new Matrix(data).transpose();
		U = Matrix.random(featureN,clusterN);
		V = Matrix.random(dataN,clusterN);
		double delta = 0.000000001;
		
		for(int k=0; k<iterN; k++){
			//System.out.println("iterater "+k+" times.");
			Matrix XV = X.times(V);
			//Matrix UVV = (U.times(V.transpose())).times(V);
			Matrix UVV = U.times(V.transpose().times(V));
			
			for(int i =0; i<featureN; i++)
				for(int j=0; j<clusterN; j++) {
					U.set(i, j, U.get(i, j)*(XV.get(i, j) + delta)/(UVV.get(i, j) + delta));
				}
			
			Matrix XU = X.transpose().times(U);
			//Matrix VUU = (V.times(U.transpose())).times(U);
			Matrix VUU = V.times(U.transpose().times(U));
			
			for(int i=0; i<dataN; i++)
				for(int j=0; j<clusterN; j++) {
					V.set(i, j, V.get(i, j)*(XU.get(i, j) + delta)/(VUU.get(i, j) + delta));
				}
			
			
		}
		
		
		//normalize
		for(int j=0; j<clusterN; j++){
			double sum = 0;
			for(int i=0; i<featureN; i++) sum += U.get(i, j)*U.get(i, j);
			sum = Math.sqrt(sum);
			for(int i=0; i<dataN; i++) if (V.get(i,j) != 0)V.set(i, j, V.get(i, j) * sum);
			for(int i=0; i<featureN; i++) if(U.get(i, j) != 0) U.set(i, j, U.get(i, j)/sum);
		}
		 
		
		//
		for(int i=0; i<dataN; i++){
			double max = 0;
			int maxj = -1;
			for(int j=0; j<clusterN; j++) if (V.get(i, j) > max){
				max = V.get(i, j);
				maxj = j;
			}
			datatag[i][1] = maxj;
		}
		
	}
	public static void main(String[] args) throws IOException{
		//test();
		Setting.fileInput(data, datatag);
		double minerror = 999999999.9;
		for (int i=0; i<10; i++){
			System.out.println((i+1)+"'st.");
			nmf(100);
			Matrix E = X.minus(U.times(V.transpose()));
			double error = E.normF();
			if (error < minerror){
				minerror = error;
				for (int j=0; j<dataN; j++) 
					for (int k = 0 ; k < 2; k ++) realtag[j][k] = datatag[j][k];
			}
		}
		Validation.init(realtag);
		System.out.println("objfunc is:" + minerror);
		System.out.println("purity is: " + Validation.purity());
		System.out.println("gini is:   " + Validation.gini());
		//output("nmf_result.txt");
		
	}
}
