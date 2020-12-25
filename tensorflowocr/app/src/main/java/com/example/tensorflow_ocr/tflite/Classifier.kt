package com.example.tensorflow_ocr.tflite

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel




import java.nio.Buffer



//class Classifier(assetManager: AssetManager, modelPath: String,inputSize: Int) {

class OCRClassifier(private val context: Context){
    private var interpreter: Interpreter
    private var labelList: List<String>
    val modelPath = "ocrfloat16.tflite"
    val labelPath = "labels.txt"
    private var inputImageWidth: Int = 200// will be inferred from TF Lite model
    private var inputImageHeight: Int = 31 // will be inferred from TF Lite model
    private var modelInputSize: Int = 4 * inputImageWidth * inputImageHeight * 1

    init {
        val assetManager = context.assets
        modelInputSize = 4 * inputImageWidth * inputImageHeight * 1
        val options = Interpreter.Options()
        options.setNumThreads(5)
        options.setUseNNAPI(true)
        interpreter = Interpreter(loadModelFile(assetManager, modelPath), options)
        labelList = assetManager.open(labelPath)
            .bufferedReader()
            .useLines { it.toList() }

    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Returns the result after running the recognition with the help of interpreter
     * on the passed bitmap
     */
    fun recognizeocrString(bitmap: Bitmap): String {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)
        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
        val result = Array(1) {LongArray(48) }
        interpreter.run(byteBuffer, result)
        return getStringResult(result)
    }

    private fun getStringResult(result: Array<LongArray>): String {
        var resultString = "";
        for(i in result[0])
        {
            Log.d("output",i.toString())
            if (i.toInt() != -1){
                Log.d("output",labelList[i.toInt()])
                resultString += labelList[i.toInt()].toString()
            }
        }
        return resultString;
    }

    private fun getBitmap(buffer: Buffer, width: Int, height: Int): Bitmap {
        buffer.rewind()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF)
            val g = (pixelValue shr 8 and 0xFF)
            val b = (pixelValue and 0xFF)

            // Convert RGB to grayscale and normalize pixel value to [0..1]
            val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
            byteBuffer.putFloat(normalizedPixelValue)
        }

        return byteBuffer
    }
    fun close() {
            interpreter?.close()
            Log.d("closing", "Closed TFLite interpreter.")
        }
}

