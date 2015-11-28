package Algorithm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Setting {
	public static int clusterN = 10;
	public static int featureN = 784;
	public static int dataN = 10000;
	
	
	public static void fileInput(double[][] data,int[][] datatag) throws IOException{
		System.out.print("input file...	");
		File file = new File("mnist.txt");
		BufferedReader f = new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF-8"));

		int lineno = 0;
		String line;
		while((line = f.readLine()) != null){
			Pattern pattern = Pattern.compile("[0-9.]+");
			Matcher matcher = pattern.matcher(line);
			int itemno = 0;
			while(matcher.find()){
				if (itemno == featureN) datatag[lineno][0] = new Integer(matcher.group(0));
				else data[lineno][itemno] = new Double(matcher.group(0));
				itemno ++;
			}	
			lineno ++;
		}
		f.close();
		System.out.println("[done]");
	}
	
}
