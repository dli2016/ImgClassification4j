package org.cripac.isee;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * Object detetion by call pyTorch ssd.
 *
 */
public class App 
{
    static {
        Loader.load(opencv_core.class);
    }
    public static void main( String[] args ) throws Exception
    {
        if (args.length != 2) {
            System.out.println("Error: BAD number of input args!");
            System.out.println("Info: 2 args is needed.");
            return;
        }
        //System.out.println("Performing validness test...");
        //System.out.println("Native library path: " + 
        //    System.getProperty("java.library.path"));
        System.out.println("Creating Classifier...");

        Classifier classifier = new ImgClassification(
            new File("src/native/test/weights/squeezenet1_1-f364aa15.pth"),
            new File("src/native/test/data/labels.json")
        );
        System.out.println("Create object classifier SUCCESSFULLY!");

        // Load image
        String image_filename = args[0];
        System.out.println(image_filename);
        opencv_core.Mat img = imread(image_filename, IMREAD_COLOR);
        byte[] img_data = new byte[img.rows() * img.cols() * img.channels()];
        img.data().get(img_data);

        // classify
        final long start = System.currentTimeMillis();
        int num_loops = Integer.parseInt(args[1]);
        for (int i = 0; i < num_loops; ++i) {
            System.out.println("Loop: " + i);
            String res = classifier.process(img_data, img.rows(), 
                img.cols(), img.channels());
            System.out.println(res);
        }
        final long end = System.currentTimeMillis();
        System.out.println("Elapsed time: " + (end - start)/(num_loops*1000.0));
        // Release.
        img.release();
        classifier.free();
    }
}
