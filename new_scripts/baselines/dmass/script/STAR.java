package joshua.metrics;

// Changed PROCore.java (text normalization function) and EvaluationMetric too

import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.util.logging.Logger;
import java.util.Arrays;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

public class STAR extends EvaluationMetric
{
	private static final Logger logger = Logger.getLogger(STAR.class.getName());

	// The maximum n-gram we care about
	protected int maxGramLength;
	protected String[] srcSentences;
	protected double[] weights;
	protected HashMap<String, Integer>[][] refNgramCounts;
	protected HashMap<String, Integer>[][] srcNgramCounts;


	/*
     You already have access to these data members of the parent
     class (EvaluationMetric):
         int numSentences;
           number of sentences in the MERT set
         int refsPerSen;
           number of references per sentence
         String[][] refSentences;
           refSentences[i][r] stores the r'th reference of the i'th
           source sentence (both indices are 0-based)
	 */


	public STAR(String[] Metric_options)
	{
		int mxGrmLn = Integer.parseInt(Metric_options[0]);
		if (mxGrmLn >= 1) {
			maxGramLength = mxGrmLn;
		} else {
			logger.severe("Maximum gram length must be positive");
			System.exit(1);
		}

		try {
			loadSources(Metric_options[1]);
		} catch (IOException e) {
			logger.severe("Error loading the source sentences from " + Metric_options[2]);
			System.exit(1);
		}

		initialize(); // set the data members of the metric

	}

	protected void initialize()
	{
		metricName = "STAR";   
		toBeMinimized = false; 
		suffStatsCount = StatIndex.values().length * maxGramLength;       

		set_weightsArray();
		set_refNgramCounts();
		set_srcNgramCounts();

	}

	public double bestPossibleScore() { return 1.0; }
	public double worstPossibleScore() { return 0.0; }

	
	  /**
	   * Sets the BLEU weights for each n-gram level to uniform.
	   */
	  protected void set_weightsArray() {
	    weights = new double[1 + maxGramLength];
	    for (int n = 1; n <= maxGramLength; ++n) {
	      weights[n] = 1.0 / maxGramLength;
	    }
	  }
	  
	  
	/**
	 * Computes the sum of ngram counts in references for each sentence (storing them in
	 * <code>refNgramCounts</code>), which are used for clipping n-gram counts.
	 */
	protected void set_refNgramCounts() {
		@SuppressWarnings("unchecked")

		HashMap<String, Integer>[][] temp_HMA = new HashMap[numSentences][maxGramLength];
		refNgramCounts = temp_HMA;

		String gram = "";
		int oldCount = 0, nextCount = 0;

		for (int i = 0; i < numSentences; ++i) {
			refNgramCounts[i] = getNgramCountsArray(refSentences[i][0]);
			// initialize to ngramCounts[n] of the first reference translation...

			// ...and update as necessary from the other reference translations
			for (int r = 1; r < refsPerSen; ++r) {



				HashMap<String, Integer>[] nextNgramCounts = getNgramCountsArray(refSentences[i][r]);

				for (int n = 1; n <= maxGramLength; ++n) {

					Iterator<String> it = (nextNgramCounts[n].keySet()).iterator();


					while (it.hasNext()) {
						gram = it.next();
						nextCount = nextNgramCounts[n].get(gram);

						if (refNgramCounts[i][n].containsKey(gram)) { // update if necessary
							oldCount = refNgramCounts[i][n].get(gram);
							refNgramCounts[i][n].put(gram, oldCount + nextCount);
						} else { // add it
							refNgramCounts[i][n].put(gram, nextCount);
						}

					}

				} //for (n)

			} // for (r)

		} // for (i)

	}


	protected void set_srcNgramCounts() {
		@SuppressWarnings("unchecked")

		HashMap<String, Integer>[][] temp_HMA = new HashMap[numSentences][maxGramLength];
		srcNgramCounts = temp_HMA;

		for (int i = 0; i < numSentences; ++i) {
			srcNgramCounts[i] = getNgramCountsArray(srcSentences[i]);
		} // for (i)
	}


	//set contents of stats[] here!
	public int[] suffStats(String cand_str, int i) 
	{
		int[] stats = new int[suffStatsCount];



		HashMap<String, Integer>[] candNgramCounts = getNgramCountsArray(cand_str);

		for (int n = 1; n <= maxGramLength; ++n) {

			//ADD OPERATIONS
			HashMap cand_sub_src = substractHashMap(candNgramCounts[n], srcNgramCounts[i][n]);
			HashMap cand_and_ref_sub_src = intersectHashMap(cand_sub_src, refNgramCounts[i][n]);
			HashMap ref_sub_src = substractHashMap(refNgramCounts[i][n], srcNgramCounts[i][n]);

			stats[StatIndex.values().length * (n - 1) +  StatIndex.ADDBOTH.ordinal()] = cand_and_ref_sub_src.keySet().size();
			stats[StatIndex.values().length * (n - 1) +  StatIndex.ADDCAND.ordinal()] = cand_sub_src.keySet().size();
			stats[StatIndex.values().length * (n - 1) +  StatIndex.ADDREF.ordinal()]  = ref_sub_src.keySet().size();
	
			
			//DELETION OPERATIONS
			HashMap src_sub_cand = substractHashMap(srcNgramCounts[i][n], candNgramCounts[n], 8, 8);
			HashMap src_sub_ref  = substractHashMap(srcNgramCounts[i][n], refNgramCounts[i][n], 8, 1);
			HashMap src_sub_cand_sub_ref = intersectHashMap(src_sub_cand, src_sub_ref, 1, 1);
			
			stats[StatIndex.values().length * (n - 1) +  StatIndex.DELBOTH.ordinal()] = sumHashMapByValues(src_sub_cand_sub_ref);
			stats[StatIndex.values().length * (n - 1) +  StatIndex.DELCAND.ordinal()] = sumHashMapByValues(src_sub_cand);
			stats[StatIndex.values().length * (n - 1) +  StatIndex.DELREF.ordinal()]  = sumHashMapByValues(src_sub_ref);

			//KEEP OPERATIONS
			HashMap src_and_cand = intersectHashMap(srcNgramCounts[i][n], candNgramCounts[n], 8, 8);
			HashMap src_and_ref  = intersectHashMap(srcNgramCounts[i][n], refNgramCounts[i][n], 8, 1);
			HashMap src_and_cand_and_ref = intersectHashMap(src_and_cand, src_and_ref, 1, 1);

			stats[StatIndex.values().length * (n - 1) +  StatIndex.KEEPBOTH.ordinal()] = sumHashMapByValues(src_and_cand_and_ref);
			stats[StatIndex.values().length * (n - 1) +  StatIndex.KEEPCAND.ordinal()] = sumHashMapByValues(src_and_cand);
			stats[StatIndex.values().length * (n - 1) +  StatIndex.KEEPREF.ordinal()]  = sumHashMapByValues(src_and_ref);

			
		}

		

		int n = 1;

		//System.out.println("CAND: " + candNgramCounts[n]);
		//System.out.println("SRC:  " + srcNgramCounts[i][n]);
		//System.out.println("REF:  " + refNgramCounts[i][n]);
		
		HashMap src_and_cand = intersectHashMap(srcNgramCounts[i][n], candNgramCounts[n], 8, 8);
		HashMap src_and_ref  = intersectHashMap(srcNgramCounts[i][n], refNgramCounts[i][n], 8, 1);
		HashMap src_and_cand_and_ref = intersectHashMap(src_and_cand, src_and_ref, 1, 1);
		//System.out.println("SRC&CAND&REF : " + src_and_cand_and_ref);
		
		HashMap cand_sub_src = substractHashMap(candNgramCounts[n], srcNgramCounts[i][n]);
		HashMap cand_and_ref_sub_src = intersectHashMap(cand_sub_src, refNgramCounts[i][n]);
		//System.out.println("CAND&REF-SRC : " + cand_and_ref_sub_src);

		HashMap src_sub_cand = substractHashMap(srcNgramCounts[i][n], candNgramCounts[n], 8, 8);
		HashMap src_sub_ref  = substractHashMap(srcNgramCounts[i][n], refNgramCounts[i][n], 8, 1);
		HashMap src_sub_cand_sub_ref = intersectHashMap(src_sub_cand, src_sub_ref, 1, 1);
		//System.out.println("SRC-REF-CAND : " + src_sub_cand_sub_ref);
		

		//System.out.println("DEBUG:" +  Arrays.toString(stats));
		//System.out.println("REF-SRC: " + substractHashMap(refNgramCounts[i], srcNgramCounts[i][0], (double)refsPerSen));

		return stats;
	}


	public double score(int[] stats)
	{
		if (stats.length != suffStatsCount) {
			System.out.println("Mismatch between stats.length and suffStatsCount (" + stats.length + " vs. " + suffStatsCount + ") in NewMetric.score(int[])");
			System.exit(1);
		}

		double sc = 0.0, sc_add = 0.0, sc_del = 0.0, sc_keep = 0.0;

		for (int n = 1; n <= maxGramLength; ++n) {

			int addCandCorrectNgram = stats[StatIndex.values().length * (n - 1) +  StatIndex.ADDBOTH.ordinal()]; 
			int addCandTotalNgram   = stats[StatIndex.values().length * (n - 1) +  StatIndex.ADDCAND.ordinal()]; 
			int addRefTotalNgram    = stats[StatIndex.values().length * (n - 1) +  StatIndex.ADDREF.ordinal()];

			double prec_add_n = 0.0 ;
			if (addCandTotalNgram > 0) {
				prec_add_n = addCandCorrectNgram / (double) addCandTotalNgram;
			} 		

			double recall_add_n = 0.0 ;
			if (addRefTotalNgram > 0) {
				recall_add_n = addCandCorrectNgram / (double) addRefTotalNgram;
			} 

			double f1_add_n = meanHarmonic(prec_add_n, recall_add_n);
			
			sc += weights[n] * f1_add_n;
			sc_add += weights[n] * f1_add_n;


			int delCandCorrectNgram = stats[StatIndex.values().length * (n - 1) +  StatIndex.DELBOTH.ordinal()]; 
			int delCandTotalNgram   = stats[StatIndex.values().length * (n - 1) +  StatIndex.DELCAND.ordinal()]; 
			int delRefTotalNgram    = stats[StatIndex.values().length * (n - 1) +  StatIndex.DELREF.ordinal()];

			double prec_del_n = 0.0 ;
			if (delCandTotalNgram > 0) {
				prec_del_n = delCandCorrectNgram / (double) delCandTotalNgram;
			} 		

			double recall_del_n = 0.0 ;
			if (delRefTotalNgram > 0) {
				recall_del_n = delCandCorrectNgram / (double) delRefTotalNgram;
			} 

			double f1_del_n = meanHarmonic(prec_del_n, recall_del_n);
			
			sc += weights[n] * f1_del_n;
			sc_del += weights[n] * f1_del_n;

			
			int keepCandCorrectNgram = stats[StatIndex.values().length * (n - 1) +  StatIndex.KEEPBOTH.ordinal()]; 
			int keepCandTotalNgram   = stats[StatIndex.values().length * (n - 1) +  StatIndex.KEEPCAND.ordinal()]; 
			int keepRefTotalNgram    = stats[StatIndex.values().length * (n - 1) +  StatIndex.KEEPREF.ordinal()];

			double prec_keep_n = 0.0 ;
			if (keepCandTotalNgram > 0) {
				prec_keep_n = keepCandCorrectNgram / (double) keepCandTotalNgram;
			} 		

			double recall_keep_n = 0.0 ;
			if (keepRefTotalNgram > 0) {
				recall_keep_n = keepCandCorrectNgram / (double) keepRefTotalNgram;
			} 

			double f1_keep_n = meanHarmonic(prec_keep_n, recall_keep_n);

			sc += weights[n] * f1_keep_n;
			sc_keep += weights[n] * f1_keep_n;
		}
		
		sc = sc / 3.0;
		sc_add = sc_add / 3.0;
		sc_del = sc_del / 3.0;
		sc_keep = sc_keep / 3.0;
		//
		//
		// set sc here!
		//
		//
    System.out.println("F1 Add = " + sc_add);
    System.out.println("F1 Del = " + sc_del);
    System.out.println("F1 Keep = " + sc_keep);
		return sc;
	}

	public double meanHarmonic(double  precision, double recall) {
		
		if (precision > 0 && recall > 0){
			return (2.0 * precision * recall) / (precision + recall);
		}
		return 0.0;
	}

	

	public void loadSources(String filepath) throws IOException {
		srcSentences = new String[numSentences];
		//BufferedReader br = new BufferedReader(new FileReader(filepath));
	    InputStream inStream = new FileInputStream(new File(filepath));
	    BufferedReader br = new BufferedReader(new InputStreamReader(inStream, "utf8"));
	      
		String line;
		int i = 0;
		while (i < numSentences && (line = br.readLine()) != null) {
			srcSentences[i] = line.trim();
			i++;
		}
		br.close();
	}
	
	public int sumHashMapByValues(HashMap<String, Integer> counter) {
		int sumcounts = 0;
		
		for (Map.Entry<String, Integer> e :  counter.entrySet()) {
			sumcounts += (int) e.getValue();
		}
		
		return sumcounts;
	}

	public HashMap<String, Integer> substractHashMap(HashMap<String, Integer> counter1, HashMap<String, Integer> counter2) {
		HashMap<String, Integer> newcounter = new HashMap<String, Integer>();

		for (Map.Entry<String, Integer> e :  counter1.entrySet()) {
			String ngram = e.getKey();
			int count1 = e.getValue();
			int count2 = counter2.containsKey(ngram) ? counter2.get(ngram) : 0;
			if (count2 == 0) {
				newcounter.put(ngram, 1);
			}
		}	  

		return newcounter;	}


	// HashMap result = counter1*ratio1 - counter2*ratio2
	public HashMap<String, Integer> substractHashMap(HashMap<String, Integer> counter1, HashMap<String, Integer> counter2, int ratio1, int ratio2) {
		HashMap<String, Integer> newcounter = new HashMap<String, Integer>();

		for (Map.Entry<String, Integer> e :  counter1.entrySet()) {
			String ngram = e.getKey();
			int count1 = e.getValue();
			int count2 = counter2.containsKey(ngram) ? counter2.get(ngram) : 0;
			int newcount = count1 * ratio1  - count2 * ratio2;
			if (newcount > 0) {
				newcounter.put(ngram, newcount);
			}
		}	  

		return newcounter;
	}


	public HashMap<String, Integer> intersectHashMap(HashMap<String, Integer> counter1, HashMap<String, Integer> counter2) {
		HashMap<String, Integer> newcounter = new HashMap<String, Integer>();

		for (Map.Entry<String, Integer> e :  counter1.entrySet()) {
			String ngram = e.getKey();
			int count1 = e.getValue();
			int count2 = counter2.containsKey(ngram) ? counter2.get(ngram) : 0;
			if (count2 > 0) {
				newcounter.put(ngram, 1);
			}
		}	  

		return newcounter;	
	}


	// HashMap result = (counter1*ratio1) & (counter2*ratio2)
	public HashMap<String, Integer> intersectHashMap(HashMap<String, Integer> counter1, HashMap<String, Integer> counter2, int ratio1, int ratio2) {
		HashMap<String, Integer> newcounter = new HashMap<String, Integer>();

		for (Map.Entry<String, Integer> e :  counter1.entrySet()) {
			String ngram = e.getKey();
			int count1 = e.getValue();
			int count2 = counter2.containsKey(ngram) ? counter2.get(ngram) : 0;
			int newcount = Math.min(count1 * ratio1, count2 * ratio2);
			if (newcount > 0) {
				newcounter.put(ngram, newcount);
			}
		}	  

		return newcounter;
	}

	protected int wordCount(String cand_str) {
		if (!cand_str.equals("")) {
			return cand_str.split("\\s+").length;
		} else {
			return 0;
		}
	}

	public HashMap<String, Integer>[] getNgramCountsArray(String cand_str) {
		if (!cand_str.equals("")) {
			return getNgramCountsArray(cand_str.split("\\s+"));
		} else {
			return getNgramCountsArray(new String[0]);
		}
	}

	public HashMap<String, Integer>[] getNgramCountsArray(String[] words) {
		@SuppressWarnings("unchecked")
		HashMap<String, Integer>[] ngramCountsArray = new HashMap[1 + maxGramLength];
		ngramCountsArray[0] = null;
		for (int n = 1; n <= maxGramLength; ++n) {
			ngramCountsArray[n] = new HashMap<String, Integer>();
		}

		int len = words.length;
		String gram;
		int st = 0;

		for (; st <= len - maxGramLength; ++st) {

			gram = words[st];
			if (ngramCountsArray[1].containsKey(gram)) {
				int oldCount = ngramCountsArray[1].get(gram);
				ngramCountsArray[1].put(gram, oldCount + 1);
			} else {
				ngramCountsArray[1].put(gram, 1);
			}

			for (int n = 2; n <= maxGramLength; ++n) {
				gram = gram + " " + words[st + n - 1];
				if (ngramCountsArray[n].containsKey(gram)) {
					int oldCount = ngramCountsArray[n].get(gram);
					ngramCountsArray[n].put(gram, oldCount + 1);
				} else {
					ngramCountsArray[n].put(gram, 1);
				}
			} // for (n)

		} // for (st)

		// now st is either len-maxGramLength+1 or zero (if above loop never entered, which
		// happens with sentences that have fewer than maxGramLength words)

		for (; st < len; ++st) {

			gram = words[st];
			if (ngramCountsArray[1].containsKey(gram)) {
				int oldCount = ngramCountsArray[1].get(gram);
				ngramCountsArray[1].put(gram, oldCount + 1);
			} else {
				ngramCountsArray[1].put(gram, 1);
			}

			int n = 2;
			for (int fin = st + 1; fin < len; ++fin) {
				gram = gram + " " + words[st + n - 1];

				if (ngramCountsArray[n].containsKey(gram)) {
					int oldCount = ngramCountsArray[n].get(gram);
					ngramCountsArray[n].put(gram, oldCount + 1);
				} else {
					ngramCountsArray[n].put(gram, 1);
				}
				++n;
			} // for (fin)

		} // for (st)

		return ngramCountsArray;

	}

	public HashMap<String, Integer> getNgramCountsAll(String cand_str) {
		if (!cand_str.equals("")) {
			return getNgramCountsAll(cand_str.split("\\s+"));
		} else {
			return getNgramCountsAll(new String[0]);
		}
	}

	public HashMap<String, Integer> getNgramCountsAll(String[] words) {
		HashMap<String, Integer> ngramCountsAll = new HashMap<String, Integer>();

		int len = words.length;
		String gram;
		int st = 0;

		for (; st <= len - maxGramLength; ++st) {

			gram = words[st];
			if (ngramCountsAll.containsKey(gram)) {
				int oldCount = ngramCountsAll.get(gram);
				ngramCountsAll.put(gram, oldCount + 1);
			} else {
				ngramCountsAll.put(gram, 1);
			}

			for (int n = 2; n <= maxGramLength; ++n) {
				gram = gram + " " + words[st + n - 1];
				if (ngramCountsAll.containsKey(gram)) {
					int oldCount = ngramCountsAll.get(gram);
					ngramCountsAll.put(gram, oldCount + 1);
				} else {
					ngramCountsAll.put(gram, 1);
				}
			} // for (n)

		} // for (st)

		// now st is either len-maxGramLength+1 or zero (if above loop never entered, which
		// happens with sentences that have fewer than maxGramLength words)

		for (; st < len; ++st) {

			gram = words[st];
			if (ngramCountsAll.containsKey(gram)) {
				int oldCount = ngramCountsAll.get(gram);
				ngramCountsAll.put(gram, oldCount + 1);
			} else {
				ngramCountsAll.put(gram, 1);
			}

			int n = 2;
			for (int fin = st + 1; fin < len; ++fin) {
				gram = gram + " " + words[st + n - 1];

				if (ngramCountsAll.containsKey(gram)) {
					int oldCount = ngramCountsAll.get(gram);
					ngramCountsAll.put(gram, oldCount + 1);
				} else {
					ngramCountsAll.put(gram, 1);
				}
				++n;
			} // for (fin)

		} // for (st)

		return ngramCountsAll;

	}


	public void printDetailedScore_fromStats(int[] stats, boolean oneLiner)
	{
		System.out.println(metricName + " = " + score(stats));


		//  for (Map.Entry<String, Integer> entry : refNgramCounts.) {
		//	    System.out.println(entry.getKey()+" : "+ entry.getValue());
		//	}
		//
		//
		// optional (for debugging purposes)
		//
		//
	}

	private enum StatIndex {KEEPBOTH, KEEPCAND, KEEPREF, DELBOTH, DELCAND, DELREF, ADDBOTH, ADDCAND, ADDREF};


}

