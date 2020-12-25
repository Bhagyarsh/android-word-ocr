package com.example.tensorflow_ocr


import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.*
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.bumptech.glide.Glide
import com.example.tensorflow_ocr.tflite.OCRClassifier
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.theartofdev.edmodo.cropper.CropImage
import com.theartofdev.edmodo.cropper.CropImageView
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc


class MainActivity : AppCompatActivity()  {
    lateinit var doc_image_view:ImageView;
    lateinit var imageView:ImageView;
    lateinit var bottomSheetButton: FrameLayout;
    lateinit var progressBar: ProgressBar;
    lateinit var ocr_text_view:TextView;
    lateinit var grayscale:Button;
    lateinit var histogram:Button;
    lateinit var black_threshold:Button;
    lateinit var white_threshold:Button;
    lateinit var histogramColor:Button;
    private lateinit var classifier: OCRClassifier
    lateinit var bottom_sheet_button_image:ImageView;
    lateinit var originImage:Uri;
    lateinit var originImageBitmap: Bitmap;
    private fun initClassifier() {
        classifier = OCRClassifier(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setSupportActionBar(findViewById(R.id.toolbar))
        originImage = Uri.EMPTY
        bottomSheetButton = findViewById(R.id.bottom_sheet_button)
        imageView = findViewById(R.id.ocr_image_view)
        histogramColor = findViewById(R.id.button_histogram_color)
        doc_image_view = findViewById(R.id.doc_image_view)
        bottom_sheet_button_image = findViewById(R.id.bottom_sheet_button_image)
        progressBar = findViewById(R.id.bottom_sheet_button_progressbar)
        ocr_text_view = findViewById(R.id.ocr_text_view)



        grayscale = findViewById(R.id.button_gray)

        histogram = findViewById(R.id.button_histogram)
        black_threshold = findViewById(R.id.button_black_threshhold)
        white_threshold = findViewById(R.id.button_white_threshhold)

        histogram.setOnClickListener{
            if (Uri.EMPTY != originImage){
                ImageHistogramGray(originImageBitmap)
            }
            else {
                Toast.makeText(this, " Please Select Image First", Toast.LENGTH_LONG).show()
            }
        }


        initClassifier()
        if (!OpenCVLoader.initDebug())
            Log.e("OpenCv", "Unable to load OpenCV");
        else
            Log.d("OpenCv", "OpenCV loaded");


        bottomSheetButton.setOnClickListener{
            CropImage.activity()
                .setAspectRatio(200, 31)
                .setGuidelines(CropImageView.Guidelines.ON)
                .start(this);
        }

        grayscale.setOnClickListener{
            if (Uri.EMPTY != originImage){
                ImageToGray(originImageBitmap)
            }
            else {
                Toast.makeText(this, "Select Image", Toast.LENGTH_LONG).show()
            }
        }
        black_threshold.setOnClickListener{

            if (Uri.EMPTY != originImage){
                ImageToblackThreshold(originImageBitmap)
            }
            else {
                Toast.makeText(this, "Select Image", Toast.LENGTH_LONG).show()
            }
        }
        white_threshold.setOnClickListener{
            if (Uri.EMPTY != originImage){
                ImagetowhiteThreshold(originImageBitmap)
            }
            else {
                Toast.makeText(this, "Select Image", Toast.LENGTH_LONG).show()
            }
        }
        histogramColor.setOnClickListener{
            if (Uri.EMPTY != originImage){
                ImagehistogramColor(originImageBitmap)
            }
            else {
                Toast.makeText(this, "Select Image", Toast.LENGTH_LONG).show()
            }
        }

    }

@RequiresApi(Build.VERSION_CODES.O)
override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    when(requestCode){
        CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE -> {
            val result = CropImage.getActivityResult(data)
            if (resultCode == Activity.RESULT_OK) {
                result.originalUri?.let { uri ->
                    originImage = uri
                    originImageBitmap = UriToBitmap(uri)
                    Glide.with(this).load(originImage).into(doc_image_view)

                }
                result.uri?.let { uri ->


                    val bitmap: Bitmap = UriToBitmap(uri)

                    analyseImage(bitmap)
                }

            }
        }
    }
}

    private  fun UriToBitmap(uri: Uri):Bitmap{
        if(Build.VERSION.SDK_INT < 28) {
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            return bitmap;

        } else {
            val source = ImageDecoder.createSource(this.contentResolver, uri)
            return ImageDecoder.decodeBitmap(source){ decoder, _, _ ->
                decoder.isMutableRequired = true

            }
        }
    }
    @RequiresApi(Build.VERSION_CODES.O)
    private fun analyseImage(decodeBitmap: Bitmap) {
        if (decodeBitmap == null){
            Toast.makeText(this, "there was an error ", Toast.LENGTH_LONG).show()
            return;
        }
      imageView.setImageBitmap(null);
        val bottomSheetBehavior = BottomSheetBehavior.from(findViewById<ListView>(R.id.bottom_sheet))
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_COLLAPSED
        showProgress();
        classifyDrawing(decodeBitmap.copy(Bitmap.Config.RGBA_F16, true))

    }

    private fun classifyDrawing(bitmap: Bitmap) {

        val result = classifier.recognizeocrString(bitmap)
        Log.d("result", result)
        ocr_text_view.text = result.toString()
    }

    private fun showProgress() {
        progressBar.visibility = View.GONE
        bottom_sheet_button_image.visibility = View.VISIBLE
    }

    private fun hideProgress() {
        progressBar.visibility = View.VISIBLE
        bottom_sheet_button_image.visibility = View.GONE
    }
    override fun onDestroy() {
        classifier.close()
        super.onDestroy()
    }

    private  fun ImageToGray(originImage: Bitmap){
        val processBitmap:Bitmap = originImage.copy(originImage.config, true)
        val tmp = Mat(processBitmap.width, processBitmap.height, CvType.CV_8UC1)
        Utils.bitmapToMat(processBitmap, tmp)
        Imgproc.cvtColor(tmp, tmp, Imgproc.COLOR_RGB2GRAY)
        Imgproc.cvtColor(tmp, tmp, Imgproc.COLOR_GRAY2RGB, 4)
        Utils.matToBitmap(tmp, processBitmap)
        Glide.with(this).load(processBitmap).into(doc_image_view)
    }


    private  fun ImageHistogramGray(originImage: Bitmap){
        val processBitmap:Bitmap = originImage.copy(originImage.config, true)
        val tmp = Mat(processBitmap.width, processBitmap.height, CvType.CV_8UC1)
        Utils.bitmapToMat(originImage, tmp)
        Imgproc.cvtColor(tmp, tmp, Imgproc.COLOR_RGB2GRAY)
        Imgproc.equalizeHist(tmp, tmp); //sr
        Utils.matToBitmap(tmp, processBitmap)
        Glide.with(this).load(processBitmap).into(doc_image_view)

    }
    private fun ImagehistogramColor(originImage: Bitmap) {
        val processBitmap:Bitmap = originImage.copy(originImage.config, true)
        val tmp = Mat(processBitmap.width, processBitmap.height, CvType.CV_8UC4)
        val filterImage =  Mat(processBitmap.width, processBitmap.height, CvType.CV_8UC4)
        Imgproc.cvtColor(tmp, filterImage, Imgproc.COLOR_RGB2HSV, 4)
        val filterImageList: MutableList<Mat> = ArrayList(3)
        Core.split(filterImage, filterImageList)
        val luminance = filterImageList[0]
        Imgproc.equalizeHist(luminance, luminance)
        filterImageList[0] = luminance
        Core.merge(filterImageList, filterImage);
        Imgproc.cvtColor(filterImage, tmp, Imgproc.COLOR_YCrCb2BGR, 4);
        Glide.with(this).load(processBitmap).into(doc_image_view)
    }

    private  fun ImageToblackThreshold(bitmap: Bitmap){
       val processBitmap =  DoThresholdGray(bitmap,false)
        Glide.with(this).load(processBitmap).into(doc_image_view)
    }
    private  fun ImagetowhiteThreshold(bitmap: Bitmap,){
        val processBitmap =  DoThresholdGray(bitmap,true)
        Glide.with(this).load(processBitmap).into(doc_image_view)
    }

    fun DoThresholdGray(BitmapGrayscale: Bitmap,White:Boolean): Bitmap? {
        val h = BitmapGrayscale.height
        val w = BitmapGrayscale.width
        val threshold: Int = 128
        val BitmapBiner = Bitmap.createBitmap(BitmapGrayscale)
        for (x in 0 until w) {
            for (y in 0 until h) {
                val pixel = BitmapGrayscale.getPixel(x, y)
                val gray: Int = Color.red(pixel)
                if (!White){
                    if (gray < threshold) {
                        BitmapBiner.setPixel(x, y, -0x1000000)
                    } else {
                        BitmapBiner.setPixel(x, y, -0x1)
                    }
                }
                else {
                    if (gray > threshold) {
                        BitmapBiner.setPixel(x, y, -0x1000000)
                    } else {
                        BitmapBiner.setPixel(x, y, -0x1)
                    }
                }

            }
        }
        return BitmapBiner
    }
}