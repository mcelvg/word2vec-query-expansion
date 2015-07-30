package com.radialpoint.word2vec;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class DistanceDemo
{
    private static final int DEFAULT_NEIGHBORHOOD = 40;

    public static void main(final String[] args) throws ClassNotFoundException,
        IOException, VectorsException
    {

        if (args.length < 1 || args.length > 2)
        {
            System.err
                .println("Usage: path/to/word2vec_model [N-neighbors]");
            System.exit(1);
        }

        int N = DEFAULT_NEIGHBORHOOD;
        if (args.length == 2)
        {
            try
            {
                N = Integer.parseInt(args[1]);
            }
            catch (final NumberFormatException nfe)
            {
                System.err
                    .println("Usage: path/to/binary_word2vec_model [N-neighbors]");
                System.exit(1);
            }
        }

        final Vectors model = ConvertVectors.loadGoogleBinary(args[0]);

        final InputStream in = System.in;
        final String prompt = "\nEnter a word or short phrase (EXIT to break): ";
        System.out.print(prompt);

        final BufferedReader br = new BufferedReader(new InputStreamReader(in,
            Charset.forName("UTF-8")));

        String line = null;
        while ((line = br.readLine()) != null && !"EXIT".equals(line))
        {
            List<Integer> ids = new ArrayList<Integer>();
            Integer index = model.getIndexOrNull(line);
            if (index != null)
            {
                ids.add(index);
            }
            if (index == null)
            {
                for (String token : line.split("\\s+"))
                {
                    index = model.getIndexOrNull(token);
                    if (index != null)
                    {
                        ids.add(index);
                    }
                }
            }
            if (ids.isEmpty())
            {
                System.out.println("\nOut of dictionary word!");
                continue;
            }
            printResult(line, ids, model, N);
            System.out.print(prompt);
        }
    }

    private static void printResult(final String input,
                                    final List<Integer> ids,
                                    final Vectors model, final int k)
    {
        final TreeSet<ScoredTerm> similarTerms =
                getNNearestNeighbors(model, ids, k);

        for (Integer id : ids)
        {
            System.out.println(String
                .format("\nWord: %s  "
                        + "Position in vocabulary: %d", model.getTerm(id), id));
        }

        System.out
            .println("\n                                      Related Term            Cosine Score");
        System.out
            .println("----------------------------------------------------------------------------");
        System.out.flush();
        for (final ScoredTerm result : similarTerms)
        {
            System.out.println(String.format("%50s%22.6f",
                                             result.term, result.score));
        }
    }

    private static TreeSet<ScoredTerm> getNNearestNeighbors(final Vectors model,
                                                            final List<Integer> ids,
                                                            final int k)
    {
        float[] target = null;

        if (ids.size() == 1)
        {
            target = model.getVector(ids.get(0));
        }
        else if (ids.size() > 1)
        {
            target = composeUnitVector(model, ids);
        }
        if (target == null)
        {
            return null;
        }
        return findNearestNeighbors(model, target, ids, k);
    }

    private static TreeSet<ScoredTerm> findNearestNeighbors(final Vectors model,
                                                            final float[] target,
                                                            final List<Integer> ids,
                                                            final int k)
    {
        final TreeSet<ScoredTerm> result = new TreeSet<ScoredTerm>();

        final int[] pos = new int[k];
        final double[] score = new double[k];

        // initialize array to hold k-nearest neighbors
        for (int i = 0; i < k; i++)
        {
            score[i] = Double.MIN_VALUE;
        }
        for (int i = 0; i < model.getVectors().length; i++)
        {
            if (arrayContains(pos, i) || ids.contains(i))
            {
                continue;
            }
            double dotproduct = 0.;
            for (int d = 0; d < target.length; ++d)
            {
                dotproduct += target[d] * model.getVector(i)[d];
            }

            for (int j = 0; j < k; j++)
            {
                if (dotproduct > score[j])
                {
                    for (int d = k - 1; d > j; d--)
                    {
                        score[d] = score[d - 1];
                        pos[d] = pos[d - 1];
                    }
                    score[j] = dotproduct;
                    pos[j] = i;
                    break;
                }
            }
        }
        for (int i = 0; i < k; i++)
        {
            final String term = model.getTerm(pos[i]);
            result.add(new ScoredTerm(score[i], term));
        }
        return result;
    }

    private static boolean arrayContains(final int[] index, final int pos)
    {
        for (int i = 0; i < index.length; i++)
        {
            if (index[i] == pos)
            {
                return true;
            }
        }
        return false;
    }

    private static float[] composeUnitVector(final Vectors model,
                                             final List<Integer> ids)
    {
        float[] sum = null;
        for (final Integer id : ids)
        {
            final float[] vec = model.getVector(id);
            if (sum == null)
            {
                sum = vec;
            }
            else
            {
                double len = 0;
                for (int i = 0; i < sum.length; ++i)
                {
                    sum[i] += vec[i];
                    len += (sum[i] * sum[i]);
                }
                len = Math.sqrt(len);
                for (int i = 0; i < sum.length; ++i)
                {
                    sum[i] /= len;
                }
            }
        }
        return sum;
    }

    private static class ScoredTerm implements Comparable<ScoredTerm>
    {
        private final double score;
        private final String term;

        ScoredTerm(double score, String term)
        {
            this.term = term.toString();
            this.score = score;
        }

        @Override
        public int compareTo(ScoredTerm othr)
        {
            int res = Double.compare(othr.score, this.score);
            if (res == 0)
            {
                res = this.term.compareTo(othr.term);
            }
            return res;
        }
    }
}
