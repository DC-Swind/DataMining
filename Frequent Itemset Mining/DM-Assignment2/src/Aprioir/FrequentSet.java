package Aprioir;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;

public class FrequentSet {
	int k;
	int minsup = 144;
	Transaction fs;
	int fs_n = 0;
	FrequentSet(int kk){
		k = kk;
		fs = new Transaction();
	}
	
	public void add(Transaction fs_tmp){
		fs.add(fs_tmp);
	}

	public void hashCount(Transaction t) {
		fs.hashCount(t,0,1,k,fs);
	}

	public void dfsTree(FrequentSet rtfs, Transaction node){
		if (node.count >= minsup){
			rtfs.add(node);
			rtfs.fs_n++;
			Apriori.FS_ALL.add(node);
			Apriori.FS_ALL.fs_n++;
			return;
		}
		for(int i=0; i<fs.itemN; i++)if(node.child[i] != null)
			dfsTree(rtfs,node.child[i]);
	}
	public FrequentSet determined() {
		//遍历树，保留大于阈值的
		FrequentSet rtfs = new FrequentSet(k);
		dfsTree(rtfs,fs);
		if(rtfs.fs_n > 0) Apriori.FS_ALL.k = rtfs.k;
		return rtfs;
	}

	public void dfsGen(FrequentSet gen,Transaction node,int deep){
		if(deep == k-1){
			//System.out.println("dot");
			for(int i=0; i<node.itemN; i++) if(node.child[i] != null)
				for(int j=i+1; j<node.itemN; j++) if(node.child[j] != null){
					//System.out.println("dot");
					Transaction g = new Transaction();
					for(int ii=0; ii<g.itemN; ii++)
						g.items[ii] = node.child[i].items[ii] | node.child[j].items[ii];
					gen.add(g);
				}
				
			return ;
		}
		for(int i=0; i<node.itemN; i++)if(node.child[i] != null)
			dfsGen(gen,node.child[i],deep+1);
	}
	public FrequentSet generate() {
		FrequentSet gen = new FrequentSet(k+1);
		dfsGen(gen,fs,0);
		return gen;
	}

	public boolean have(Transaction son){
		Transaction node = fs;
		for(int i=0; i<son.itemN; i++) if(son.items[i] == 1){
			node = node.child[i];
			if(node == null) return false;
		}
		return true;
	}
	public void dfsPrune(FrequentSet fsk,Transaction node,int deep){
		if (deep == k - 1){
			for(int ii =0; ii<node.itemN; ii++) if (node.child[ii] != null){
				Transaction son = new Transaction();
				for(int i=0; i<son.itemN; i++) son.items[i] = node.child[ii].items[i];
				for(int i=0; i<son.itemN; i++) if (son.items[i] == 1){
					son.items[i] = 0;
					if (!fsk.have(son)){
						node.child[ii] = null;
						break;
					}
					son.items[i] = 1;
				}
			}
			return ;
		}
		for(int i=0; i<node.itemN; i++) if(node.child[i] != null)
			dfsPrune(fsk,node.child[i],deep+1);
	}
	public void prune(FrequentSet fsk) {
		dfsPrune(fsk,fs,0);
		
	}
	
	public void dfsPrint(Transaction node,int deep,BufferedWriter f) throws IOException{
		DecimalFormat df=new DecimalFormat("0.000");
		if(node.count > 0){
			//for(int i=0; i<node.itemN; i++) if(node.items[i] ==1) System.out.print((i+1)+" ");
			//System.out.println(df.format((double)node.count/(double)Apriori.transactionsN));
			for(int i=0; i<node.itemN; i++) if(node.items[i] == 1) f.write((i+1)+" ");
			f.write(df.format((double)node.count/(double)Apriori.transactionsN) + "\n");
		}
		if (deep == k){
			return ;
		}
		for(int i=0; i<node.itemN; i++)if(node.child[i] != null)
			dfsPrint(node.child[i],deep+1,f);
	}
	public void print() throws IOException{
		System.out.print("print answer...	");
		File file = new File("result.txt");
		if (!file.exists()) file.createNewFile();
		BufferedWriter f = new BufferedWriter(new FileWriter(file));
		dfsPrint(fs,0,f);
		f.close();
		System.out.println("[done]");
	}
}
