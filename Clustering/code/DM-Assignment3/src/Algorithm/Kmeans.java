package Algorithm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;


public class Kmeans {
	public static int clusterN = Setting.clusterN;
	public static int featureN = Setting.featureN;
	public static int dataN = Setting.dataN;
	public static double means[][] = new double[clusterN][featureN];
	public static double newmeans[][] = new double[clusterN][featureN];
	public static int newmeansN[] = new int[clusterN];
	public static double data[][] = new double[dataN][featureN];
	public static int datatag[][] = new int[dataN][2];
	public static int realtag[][] = new int[dataN][2];
	public static double error = 0;
	
	public static double distance(double[] a,double[] b){
		double dis = 0;
		for(int i=0; i<featureN; i++){
			dis += (a[i]-b[i]) * (a[i]-b[i]);
		}
		return dis;
	}
	public static boolean check(double[] a){
		double sum = 0;
		for(int i=0; i<featureN; i++) sum+=a[i];
		if (sum == 0) return false;
		return true;
	}
	public static void assign(){
		for(int i=0; i<dataN; i++){
			double mindistance = 9999999;
			int minj = -1;
			for(int j=0; j<clusterN; j++){
				double dis = distance(data[i],means[j]);
				if (dis < mindistance){
					mindistance = dis;
					minj = j;
				}
			}
			for(int k=0; k<featureN; k++){
				newmeans[minj][k] = (newmeans[minj][k] * newmeansN[minj] + data[i][k])/(newmeansN[minj] + 1);
			}
			newmeansN[minj]++;
			datatag[i][1] = minj;
		}
		
	}
	public static void optimize(){
		
		for(int i=0; i<clusterN; i++)if(newmeansN[i] != 0){
			for(int j=0; j<featureN; j++){
				means[i][j] = newmeans[i][j];
				newmeans[i][j] = 0;
			}
			newmeansN[i] = 0;
		}
		
	}
	public static boolean equal(double[][] a,double[][] b){
		for(int i=0; i<clusterN; i++)
			for(int j=0; j<featureN; j++) if (a[i][j] != b[i][j]) return false;
		return true;
	}
	public static void init_means(double[][] means){
		Random rand = new Random();
		boolean[] flag = new boolean[dataN];
		for(int i=0; i<dataN; i++) flag[i] = true;
		for(int i=0; i<clusterN; i++){
			int index= rand.nextInt(dataN);
			while(!flag[index]){
				index = rand.nextInt(dataN);
			}
			flag[index] = false;
			for(int j=0; j<featureN; j++){ 
				means[i][j] = data[index][j];
				newmeans[i][j] = 0;
			}
			newmeansN[i] = 0;
		}
	}
	public static void kmeans() throws IOException{
		init_means(means);
		
		int iter = 1;
		do{
			assign();
			if(equal(means,newmeans)){
				//System.out.println("iterater "+iter+" times.");
				break;
			}
			optimize();
			iter++;
		}while(true);
		
		objfunc();
	}
	public static void objfunc(){
		for (int i=0; i<dataN; i++){
			double d = distance(data[i],means[datatag[i][1]]);
			error += d * d;
		}
	}
	
	public static void output(String filename) throws IOException{
		System.out.print("print answer...	");
		File file = new File(filename);
		if (!file.exists()) file.createNewFile();
		BufferedWriter f = new BufferedWriter(new FileWriter(file));
		
		for(int i=0; i<clusterN; i++){
			f.write("Cluster "+i+" :");
			for(int j=0; j<featureN; j++){
				f.write(means[i][j]+" ");
			}
			f.write("\n");;
		}
		
		f.close();
		System.out.println("[done]");
	}
	public static void main(String[] args) throws IOException{
		Setting.fileInput(data,datatag);
		double minerror = 99999999999.9;
		for (int i=0; i<10; i++){
			System.out.println((i+1) + "'st.");
			error = 0;
			kmeans();
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
		//output("result.txt");
	}
}
