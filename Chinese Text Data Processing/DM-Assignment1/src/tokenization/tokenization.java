package tokenization;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.queryParser.ParseException;
import org.apache.lucene.queryParser.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.LockObtainFailedException;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.Version;
import org.wltea.*;
import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;
import org.wltea.analyzer.lucene.IKAnalyzer;
class ff{
	TreeMap<Integer,Integer>[] post = new TreeMap[200];
	TreeMap<Integer,Double>[] tf = new TreeMap[200];
	int postn;
	int[] tokenn = new int[200];
	ff(){
		postn = 0;
	}
};
public class tokenization {
	public static ff basketball = new ff();
	public static ff computer = new ff();
	public static ff fleamarket = new ff();
	public static ff girls = new ff();
	public static ff jobexpress = new ff();
	public static ff mobile = new ff();
	public static ff stock = new ff();
	public static ff suggestions = new ff();
	public static ff warandpeace = new ff();
	public static ff football = new ff();
	public static String stopwords[] = new String[1300];
	public static int stopwordsn = 0;
	public static String wordlist[] = new String[30000];
	public static double idf[] = new double[30000];
	public static int wordn = 0;
	public static void stopwordsInput() throws IOException{
		File file = new File("Chinese-stop-words.txt");
		//BufferedReader f = new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF-8"));
		BufferedReader f = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
		String line = null;
		while((line = f.readLine()) != null){
			stopwords[stopwordsn] = new String(line);
			stopwordsn++;
		}
		f.close();
	}
	public static boolean isstopword(String word){
		for(int i=0; i < stopwordsn; i++) if (stopwords[i].compareTo(word) == 0) return true;
		
		//for(int i=0; i < stopwordsn; i++) if (word.contains(stopwords[i])) return true;
		
		Pattern pattern = Pattern.compile("[a-zA-Z0-9.\\+\\-\\/\\@,]+");   
		Matcher matcher = pattern.matcher(word);
        String ans = "";
        boolean first = true;
        while (matcher.find()){
        	ans = matcher.group(0);
        	//System.out.println(ans + " : " + word);
        	return true;
        }
		return false;
	}
	
	public static int getwordindex(String word){
		for(int i=0; i<wordn; i++) if (wordlist[i].compareTo(word) == 0) return i;
		wordlist[wordn] = word;
		wordn++;
		return wordn-1;
	}
	
	public static void fileInput(String filename,ff topic) throws IOException{
		File file = new File("lily/"+filename+".txt");
		BufferedReader f = new BufferedReader(new InputStreamReader(new FileInputStream(file),"UTF-8"));

		String line = null;
		while((line = f.readLine()) != null){
			String text = new String(line);
			StringReader sr=new StringReader(text);  
	        IKSegmenter ik=new IKSegmenter(sr, true);  
	        Lexeme lex=null;  
	        
			topic.post[topic.postn] = new TreeMap();
			topic.tf[topic.postn] = new TreeMap();
			
	        while((lex=ik.next())!=null){  
	        	String token = lex.getLexemeText();
	            if (isstopword(token)){
	            	
	            } else{
	            	topic.tokenn[topic.postn]++;
	            	int index = getwordindex(token);
	            	int count = (topic.post[topic.postn].get(index) == null)?0:topic.post[topic.postn].get(index);
	            	topic.post[topic.postn].put(index,count + 1);
	            }
	        }  
	        int i = topic.postn;
	        for (int key : topic.post[i].keySet()) {
	        	topic.tf[i].put(key, new Double(topic.post[i].get(key))/topic.tokenn[i]);
	        	//System.out.println(wordlist[key] + " " + new Double(topic.post[i].get(key))/topic.tokenn[i]);
	        }
	        topic.postn++;
		}
		f.close();
	}
	
	public static void outputtokens() throws IOException{
		File file = new File("tokenlist.txt");
		if (!file.exists()) file.createNewFile();
		BufferedWriter f = new BufferedWriter(new FileWriter(file));
		for(int i=0; i<wordn; i++){
			f.write(wordlist[i]+"\n");
			//f.newLine();
		}
		f.close();
	}

	public static void calculateIDF(){
		int N = basketball.postn + computer.postn + fleamarket.postn + girls.postn + jobexpress.postn + mobile.postn + stock.postn +
				suggestions.postn + warandpeace.postn + football.postn;
		System.out.println("the number of posts in 10 files is :" + N);
		for(int i=0; i<wordn; i++){
			int n = 0;
			for(int j=0; j<basketball.postn; j++) n += (basketball.post[j].get(i) == null)?0:1;
			for(int j=0; j<computer.postn; j++) n += (computer.post[j].get(i) == null)?0:1;
			for(int j=0; j<fleamarket.postn; j++) n += (fleamarket.post[j].get(i) == null)?0:1;
			for(int j=0; j<girls.postn; j++) n += (girls.post[j].get(i) == null)?0:1;
			for(int j=0; j<jobexpress.postn; j++) n += (jobexpress.post[j].get(i) == null)?0:1;
			for(int j=0; j<mobile.postn; j++) n += (mobile.post[j].get(i) == null)?0:1;
			for(int j=0; j<stock.postn; j++) n += (stock.post[j].get(i) == null)?0:1;
			for(int j=0; j<suggestions.postn; j++) n += (suggestions.post[j].get(i) == null)?0:1;
			for(int j=0; j<warandpeace.postn; j++) n += (warandpeace.post[j].get(i) == null)?0:1;
			for(int j=0; j<football.postn; j++) n += (football.post[j].get(i) == null)?0:1;
			idf[i] = Math.log(N/n);
		}
		
		
	}
	public static void calculateTFIDF(String filename,ff topic) throws IOException{
		File file = new File("result/"+filename+".txt");
		if (!file.exists()) file.createNewFile();
		BufferedWriter f = new BufferedWriter(new FileWriter(file));
		
		
		for(int i=0; i<topic.postn; i++){
			DecimalFormat df=new DecimalFormat(".####");
			
			
			/*
			double max = 0;
			int maxkey = -1;
			for (int key : topic.post[i].keySet()) {
	        	double tfidf = topic.tf[i].get(key) * idf[key];
	        	if (tfidf > max){
	        		max = tfidf;
	        		maxkey = key;
	        	}
	        	f.write(wordlist[key] + ":" + df.format(tfidf) + " ");
	        }
			if (maxkey >= 0 ) f.write(wordlist[maxkey] + ":" + df.format(max) + " ");
			*/
			
			for (int key : topic.post[i].keySet()) {
	        	double tfidf = topic.tf[i].get(key) * idf[key];
	        	f.write(key + ":" + df.format(tfidf) + " ");
	        }
	        
			f.newLine();
			
		}
		f.close();
	}
	public static void main(String[] args) throws IOException{
		System.out.println("start reading stopwords...");
		stopwordsInput();
		System.out.println("start calculating TF...");
		fileInput("Basketball",basketball);
		fileInput("D_Computer",computer);
		fileInput("FleaMarket",fleamarket);
		fileInput("Girls",girls);
		fileInput("JobExpress",jobexpress);
		fileInput("Mobile",mobile);
		fileInput("Stock",stock);
		fileInput("V_Suggestions",suggestions);
		fileInput("WarAndPeace",warandpeace);
		fileInput("WorldFootball",football);
		System.out.println("total tokens in 10 files :" + wordn + "  then start calculating IDF and TF-IDF...");
		calculateIDF();
		calculateTFIDF("Basketball",basketball);
		calculateTFIDF("D_Computer",computer);
		calculateTFIDF("FleaMarket",fleamarket);
		calculateTFIDF("Girls",girls);
		calculateTFIDF("JobExpress",jobexpress);
		calculateTFIDF("Mobile",mobile);
		calculateTFIDF("Stock",stock);
		calculateTFIDF("V_Suggestions",suggestions);
		calculateTFIDF("WarAndPeace",warandpeace);
		calculateTFIDF("WorldFootball",football);
		System.out.println("start write tokens to file.");
		outputtokens();
		System.out.println("program end.");
		/*
		String text="基于java语言开发的轻量级的中文分词工具包";  
        StringReader sr=new StringReader(text);  
        IKSegmenter ik=new IKSegmenter(sr, true);  
        Lexeme lex=null;  
        while((lex=ik.next())!=null){  
            System.out.print(lex.getLexemeText()+"|");  
        } 
        */ 
		/*
		//Lucene Document的域名
		String fieldName = "text";
		String text = "IK Analyzer是一个结合词典分词和文法分词的中文分词开源工具包。它使用了全新的正向迭代最细粒度切分算法。";
		
		//实例化IKAnalyzer分词器 
		Analyzer analyzer = new IKAnalyzer(); 
		Directory directory = null; 
		IndexWriter iwriter = null; 
		IndexReader ireader = null; 
		IndexSearcher isearcher = null; 
		try { 
			//建立内存索引对象 
			directory = new RAMDirectory(); 
			//配置IndexWriterConfig 
			IndexWriterConfig iwConfig = new IndexWriterConfig(Version.LUCENE_34 , analyzer); 
			iwConfig.setOpenMode(OpenMode.CREATE_OR_APPEND); 
			iwriter = new IndexWriter(directory , iwConfig);
			//写入索引 
			Document doc = new Document(); 
			doc.add(new Field("ID", "10000", Field.Store.YES, Field.Index.NOT_ANALYZED)); 
			doc.add(new Field(fieldName, text, Field.Store.YES, Field.Index.ANALYZED)); 
			iwriter.addDocument(doc); 
			iwriter.close();
			
			//搜索过程********************************** 
			//实例化搜索器
			ireader = IndexReader.open(directory);
			isearcher = new IndexSearcher(ireader); 
			String keyword = "分词"; 
			//使用QueryParser查询分析器构造Query对象 
			QueryParser qp = new QueryParser(Version.LUCENE_34, fieldName, analyzer); 
			qp.setDefaultOperator(QueryParser.AND_OPERATOR); 
			Query query = qp.parse(keyword);
			
			//搜索相似度最高的5条记录 
			TopDocs topDocs = isearcher.search(query , 5); 
			System.out.println("命中：" + topDocs.totalHits); 
			
			//输出结果 
			ScoreDoc[] scoreDocs = topDocs.scoreDocs; 
			for (int i = 0; i < topDocs.totalHits; i++){ 
				Document targetDoc = isearcher.doc(scoreDocs[i].doc); 
				System.out.println("内容：" + targetDoc.toString()); 
			}
		} catch (CorruptIndexException e) {
			e.printStackTrace(); 
		} catch (LockObtainFailedException e) {
			e.printStackTrace(); 
		} catch (IOException e) { 
			e.printStackTrace(); 
		} catch (ParseException e) { 
			e.printStackTrace(); 
		} finally{ 
			if(ireader != null){ 
				try { 
					ireader.close(); 
				} catch (IOException e) { 
					e.printStackTrace(); 
				} 
			} 
			if(directory != null){ 
				try { 
					directory.close(); 
				} catch (IOException e) { 
					e.printStackTrace(); 
				} 
			} 
		}
		*/
	}
}
