package Aprioir;

public class Transaction {
	public static int itemN = 11;
	public int[] items;
	public Transaction[] child = new Transaction[itemN];
	public int count = 0;
	Transaction(){
		items = new int[itemN];
		for (int i=0; i<itemN; i++) items[i] = 0;
	}
	
	public void add(Transaction fs){
		Transaction temp = this;
		for(int i=0; i<fs.itemN; i++) if (fs.items[i] == 1){
			if (temp.child[i] == null) temp.child[i] = new Transaction();
			temp = temp.child[i];
		}
		for(int i=0; i<fs.itemN; i++)
			temp.items[i] = fs.items[i];
		temp.count = fs.count;
	}

	public void hashCount(Transaction t, int pos, int deep,int maxdeep,Transaction node) {
		// TODO Auto-generated method stub
		if(deep > maxdeep){ 
			node.count++;
			return;
		}
		if (pos >= t.itemN) return;
		hashCount(t,pos+1,deep,maxdeep,node); //没有往下走
		if(t.items[pos] == 1 && node.child[pos] != null)
			hashCount(t,pos+1,deep+1,maxdeep,node.child[pos]); //往下走一
		
	}
}
