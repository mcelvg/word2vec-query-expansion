/*
 * Copyright 2014 Radialpoint SafeCare Inc. All Rights Reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package com.radialpoint.word2vec;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.util.Arrays;

/**
 * This program takes vectors are produced by the C program word2vec and transforms them into a Java binary file to be
 * read by the Vectors class
 */
public class ConvertVectors {

    /**
     * @param args
     *            the input C vectors file, output Java vectors file
     */
    public static void main(String[] args) throws VectorsException, IOException
    {
        File outputFile = new File(args[1]);

        Vectors instance = loadGoogleBinary(args[0]);
        
        FileOutputStream fos = new FileOutputStream(outputFile);
        instance.writeTo(fos);
    }
    
    public static Vectors loadGoogleBinary(String pathToFile) throws VectorsException, IOException
    {
        float[][] vectors;
        String[] vocabVects;
        int words;
        int size;

        File vectorFile = new File(pathToFile);

        if (!vectorFile.exists())
            throw new VectorsException("Vectors file not found");

        FileInputStream fis = new FileInputStream(vectorFile);

        BufferedInputStream in = new BufferedInputStream(fis);

        StringBuilder sb = new StringBuilder();
        char ch = (char) in.read();
        while (ch != '\n') {
            sb.append(ch);
            ch = (char) in.read();
        }

        String line = sb.toString();
        String[] parts = line.split("\\s+");
        words = (int) Long.parseLong(parts[0]);
        size = (int) Long.parseLong(parts[1]);
        vectors = new float[words][size];
        vocabVects = new String[words];

        System.out.println("" + words + " words with size " + size + " per vector.");

        byte[] bytes = new byte[4 * size];
        ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        
        for (int w = 0; w < words; w++) {
            if (w % (words / 10) == 0) {
                System.out.println("Read " + w + " words");
            }
            vocabVects[w] = readNextWord(in, Charset.defaultCharset());

            in.read(bytes);
            double len = 0;
            for (int j = 0; j < size; j++)
            {
                vectors[w][j] = buf.getFloat(j * 4);
                len += vectors[w][j] * vectors[w][j];
            }
            // convert to unit vector
            len = (float) Math.sqrt(len);
            for (int k = 0; k < size; k++)
            {
                vectors[w][k] /= len;
            }
        }
        fis.close();
        in.close();
        return new Vectors(vectors, vocabVects);
    }
    
    private static String readNextWord(BufferedInputStream in, Charset cs) throws VectorsException
    {
        // larger than necessary, word2vec.c limits max word length to 50
        byte[] buf = new byte[500];
        try
        {
            int p = 0;
            char ch = (char) in.read();
            // GoogleNews-vectors-negative300.bin dosen't include '\n' chars
            // between vectors - this check allows you to load binary files
            // created with either version of Mikolov's word2vec code
            while (Character.isWhitespace(ch))
            {
                ch = (char) in.read();
            }
            while (!Character.isWhitespace(ch))
            {
                buf[p] = (byte) ch;
                ch = (char) in.read();
                p++;
            }
            buf = Arrays.copyOf(buf, p);
        }
        catch (IOException e)
        {
            throw new VectorsException("Failed to read next word");
        }
        return new String(buf, cs);
    }

}
