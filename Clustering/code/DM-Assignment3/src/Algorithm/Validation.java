package Algorithm;
public class Validation {
	public static int clusterN = Setting.clusterN;
	public static int featureN = Setting.featureN;
	public static int dataN = Setting.dataN;
	public static int[][] m = new int[clusterN][clusterN];
	public static int[] N = new int[clusterN];
	public static int[] M = new int[clusterN];
	public static int[] P = new int[clusterN];
	public static double[] G = new double[clusterN];
	
	public static void init(int[][] datatag){
		for(int i=0; i<clusterN; i++){
			for(int j=0; j<clusterN; j++) m[i][j] = 0;
			P[i] = 0;
			N[i] = 0;
			M[i] = 0;
			G[i] = 0;
		}
		for(int i=0; i<dataN; i++) m[datatag[i][0]][datatag[i][1]]++;
		
		for(int i=0; i<clusterN; i++)
			for(int j=0; j<clusterN; j++) {
				N[i] += m[i][j];
				M[j] += m[i][j];
				if (m[i][j] > P[j]) P[j] = m[i][j];
			}
	}
	
	public static double purity(){
		double a = 0;
		double b = 0;
		for(int j=0; j<clusterN; j++){
			a += P[j];
			b += M[j];
		}
		return a/b;
	}
	
	public static double gini(){
		for(int j=0; j<clusterN; j++) {
			for(int i=0; i<clusterN; i++) G[j] += (m[i][j] == 0)?0:((double)m[i][j]/(double)M[j])*((double)m[i][j]/(double)M[j]);
			G[j] = 1 - G[j];
		}
		double a = 0;
		double b = 0;
		for(int j=0; j<clusterN; j++){
			a += G[j] * M[j];
			b += M[j];
		}
		return a/b;
	}
}
