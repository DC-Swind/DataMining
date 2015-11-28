package Aprioir;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Apriori {
	public static int itemsN = 11;
	public static int transactionsN = 1000;
	public static Transaction[] transactions = new Transaction[transactionsN];
	public static FrequentSet[] FS = new FrequentSet[itemsN];
	public static FrequentSet FS_ALL = new FrequentSet(0);
	
	public static void fileInput() throws IOException{
		System.out.print("input file...	");
		File file = new File("assignment2-data.txt");
		BufferedReader f = new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF-8"));

		int lineno = 0;
		String line = f.readLine();
		while((line = f.readLine()) != null){
			transactions[lineno] = new Transaction();
	
			Pattern pattern = Pattern.compile("[01]");
			Matcher matcher = pattern.matcher(line);
			int itemno = 0;
			while(matcher.find()){
				transactions[lineno].items[itemno] = new Integer(matcher.group(0));
				itemno ++;
			}
			
			lineno ++;
		}
		f.close();
		System.out.println("[done]");
	}

	public static void hashCounting(FrequentSet fs){
		for (int i=0; i<transactionsN; i++)
			fs.hashCount(transactions[i]);
	}
	public static void Apriori(){
		System.out.print("Apriori...	");
		FrequentSet[] CFS = new FrequentSet[itemsN];

		int k = 1;
		CFS[1] = new FrequentSet(1);
		for (int i=0; i<itemsN; i++){
			Transaction cfs = new Transaction();
			cfs.items[i] = 1;
			CFS[1].add(cfs);
		}
		hashCounting(CFS[1]);
		FS[1] = CFS[1].determined();

		while(FS[k].fs_n != 0){
			CFS[k+1] = FS[k].generate();
			//CFS[k+1].print();
			CFS[k+1].prune(FS[k]);
			hashCounting(CFS[k+1]);
			//CFS[k+1].print();
			FS[k+1] = CFS[k+1].determined();
			//FS[k+1].print();
			k++;
		}
		System.out.println("[done]");
	}
	public static void BruteForce(){
		for(int i=0; i<11; i++)
			for(int j=i+1; j<11; j++){
				int count = 0;
				for(int k=0; k<1000; k++) if (transactions[k].items[i] == 1 && transactions[k].items[j] == 1)
					count++;
				if (count >= 144) System.out.println(i + " " + j + "  |"+count);
			}
	}
	public static void main(String[] args) throws IOException{
		fileInput();
		Apriori();
		//BruteForce();
		FS_ALL.print();
	}
}
