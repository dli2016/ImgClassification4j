
package org.cripac.isee;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.CharacterCodingException;
import java.nio.file.AccessDeniedException;
import java.util.Collection;

public class ImgClassification implements Classifier {

    static {
        try {
            System.out.println("Load native libraries from " + 
                System.getProperty("java.library.path"));
            System.loadLibrary("cppwrapper_pytorch_squeezenet_jni");
            System.out.println("Native library loaded successfully!");
        } catch (Throwable t) {
            System.out.println("Failed to load native library!");
            t.printStackTrace();
        }
    }

    private long handle;
    private long dl;

    private native long loadLibrary(String lib_name);
    private native long initialize(long dl, String modelPath, String labelPath);
    private native String classify(long handle, byte[] imgBuf, int h, int w, int c);
    private native int release(long handle, long dl);
    private native void closeLibrary(long dl);

    // Constructor
    public ImgClassification(File modelFile, File labelFile) throws 
        FileNotFoundException, AccessDeniedException, CharacterCodingException {
        
        if (!modelFile.exists()) {
            throw new FileNotFoundException("Cannot find " + 
                modelFile.getPath());
        }
        if (!modelFile.canRead()) {
            throw new AccessDeniedException("Cannot read " + 
                modelFile.getPath());
        }
        if (!labelFile.exists()) {
            throw new FileNotFoundException("Cannot find " + 
                labelFile.getPath());
        }
        if (!labelFile.canRead()) {
            throw new AccessDeniedException("Cannot read " + 
                labelFile.getPath());
        }
        
        // Load library.
        dl = loadLibrary("libpytorch_squeezenet.so");
        if (dl > 0) {
            System.out.println("Load library of python code Done!");
        } else {
            System.exit(-1);
        }
        System.out.println("Initializing ...");
        // Initialize.
        handle = initialize(dl, modelFile.getPath(), labelFile.getPath());
        if (handle > 0) {
            System.out.println("Initialization Done!");
        } else {
            System.out.println("Initialize FAILED!");
            System.exit(-1);
        }
    }

    @Override
    public void free() {
        release(handle, dl);
        closeLibrary(dl);
        //super.finalize();
        System.out.println("Resources RELEASED!");
    }

    @Override
    public String process(byte[] frame, int h, int w, int c) {
        String category = classify(handle, frame, h, w, c);
        return category; // Test
    }

    @Override
    protected void finalize() throws Throwable {
        //release(handle, dl);
        //closeLibrary(dl);
        super.finalize();
        //System.out.println("Resources RELEASED!");
    }
}
