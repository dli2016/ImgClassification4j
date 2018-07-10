
package org.cripac.isee;

public interface Classifier {
    String process(byte[] frame, int h, int w, int c);
    void free();
}
