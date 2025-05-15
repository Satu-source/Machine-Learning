package com.example.food_class_app

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
// import org.tensorflow.lite.support.image.ops.Rot90Op // Uncomment if rotation handling is needed
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var captureButton: Button
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView

    // TFLite variables
    private var interpreter: Interpreter? = null
    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0
    private lateinit var labels: List<String>

    // Notification constants
    private val CHANNEL_ID = "food_classification_channel"
    private val NOTIFICATION_ID = 101

    // ActivityResultLaunchers
    private val takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        if (bitmap != null) {
            imageView.setImageBitmap(bitmap)
            classifyImage(bitmap)
        } else {
            Toast.makeText(this, "Failed to capture image", Toast.LENGTH_SHORT).show()
        }
    }

    private val requestCameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                takePictureLauncher.launch(null)
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
            }
        }

    private val requestNotificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                // Permission is granted. You might want to show a pending notification if one was queued.
                Toast.makeText(this, "Notification permission granted.", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Notification permission denied. Results won't be shown as notifications.", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main_activity)

        captureButton = findViewById(R.id.captureButton)
        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)

        initializeTFLiteModel()
        createNotificationChannel() // Create notification channel early
        askForNotificationPermission() // Ask for notification permission on Android 13+

        captureButton.setOnClickListener {
            askForCameraPermissionAndLaunchCamera()
        }
    }

    private fun initializeTFLiteModel() {
        try {
            val modelFilename = "your_model.tflite" // ** REPLACE WITH YOUR MODEL FILE NAME **
            val labelsFilename = "your_labels.txt"  // ** REPLACE WITH YOUR LABELS FILE NAME **

            interpreter = Interpreter(loadModelFile(modelFilename))
            labels = FileUtil.loadLabels(this, labelsFilename)

            val inputTensor = interpreter?.getInputTensor(0)
            inputImageWidth = inputTensor?.shape()?.get(2) ?: 224 // Default to 224
            inputImageHeight = inputTensor?.shape()?.get(1) ?: 224 // Default to 224
            Log.i("MainActivity", "TFLite Model Initialized. Input Shape: [$inputImageHeight, $inputImageWidth]")

        } catch (e: IOException) {
            Log.e("MainActivity", "Error initializing TFLite Model:", e)
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            resultTextView.text = "Error: Model not loaded."
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelFilename: String): ByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        // It's good practice to close resources, though MappedByteBuffer might handle some of this.
        // fileDescriptor.close() // Closing FD also closes underlying stream & channel for it
        // inputStream.close()
        // fileChannel.close()
        return mappedByteBuffer
    }

    private fun askForCameraPermissionAndLaunchCamera() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                takePictureLauncher.launch(null)
            }
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                // Show an educational UI to the user explaining why the permission is needed
                // For this example, a Toast is used. Consider a Dialog for better UX.
                Toast.makeText(this, "Camera permission is required to capture images for classification.", Toast.LENGTH_LONG).show()
                requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
            else -> {
                requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun classifyImage(bitmap: Bitmap) {
        if (interpreter == null) {
            Toast.makeText(this, "Model is not initialized.", Toast.LENGTH_SHORT).show()
            resultTextView.text = "Classification Error: Model not ready."
            return
        }

        try {
            var tensorImage = TensorImage(interpreter!!.getInputTensor(0).dataType())
            tensorImage.load(bitmap)

            // ** IMPORTANT: Adjust NormalizeOp based on your model's training **
            // Example: NormalizeOp(0f, 1f) if model expects 0-1 from 0-255 pixels
            // Example: NormalizeOp(127.5f, 127.5f) if model expects -1 to 1 from 0-255 pixels
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
                // .add(Rot90Op(imageOrientation / 90)) // Optional: if you need to handle image rotation
                .add(NormalizeOp(0f, 255f)) // ** ADJUST THIS NORMALIZATION **
                .build()

            tensorImage = imageProcessor.process(tensorImage)

            val outputTensor = interpreter!!.getOutputTensor(0)
            val outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

            interpreter!!.run(tensorImage.buffer, outputBuffer.buffer.rewind())

            val tensorLabel = TensorLabel(labels, outputBuffer)
            val resultsMap = tensorLabel.mapWithFloatValue

            var topResult = "Unknown"
            var maxConfidence = 0f
            if (resultsMap.isNotEmpty()) {
                val sortedResults = resultsMap.entries.sortedByDescending { it.value }
                topResult = sortedResults[0].key
                maxConfidence = sortedResults[0].value
            }
            val confidencePercentage = maxConfidence * 100

            val resultString = "Food: $topResult (Confidence: ${String.format("%.1f", confidencePercentage)}%)"
            resultTextView.text = resultString
            showNotification(resultString)

        } catch (e: Exception) {
            Log.e("MainActivity", "Error classifying image", e)
            Toast.makeText(this, "Error classifying image: ${e.message}", Toast.LENGTH_LONG).show()
            resultTextView.text = "Classification Error."
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val name = getString(R.string.notification_channel_name) // Use string resource
            val descriptionText = getString(R.string.notification_channel_description) // Use string resource
            val importance = NotificationManager.IMPORTANCE_DEFAULT
            val channel = NotificationChannel(CHANNEL_ID, name, importance).apply {
                description = descriptionText
            }
            val notificationManager: NotificationManager =
                getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun showNotification(result: String) {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        }
        val pendingIntent: PendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT // Added UPDATE_CURRENT
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification_icon) // ** CREATE THIS DRAWABLE **
            .setContentTitle("Food Classification Result")
            .setContentText(result)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)

        with(NotificationManagerCompat.from(this)) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
                ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) {
                try {
                    notify(NOTIFICATION_ID, builder.build())
                } catch (e: SecurityException) {
                    Log.e("MainActivity", "SecurityException while showing notification (missing permission?): ${e.message}")
                    Toast.makeText(this@MainActivity, "Could not show notification due to permission issue.", Toast.LENGTH_LONG).show()
                }
            } else {
                Log.w("MainActivity", "POST_NOTIFICATIONS permission not granted. Notification not shown.")
                // Optionally, inform the user why the notification isn't showing,
                // or queue it to show if permission is granted later.
            }
        }
    }

    private fun askForNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // TIRAMISU is API 33
            when {
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.POST_NOTIFICATIONS
                ) == PackageManager.PERMISSION_GRANTED -> {
                    // Permission already granted
                    Log.i("MainActivity", "Notification permission already granted.")
                }
                shouldShowRequestPermissionRationale(Manifest.permission.POST_NOTIFICATIONS) -> {
                    Toast.makeText(this, "Notification permission is needed to display classification results in the status bar.", Toast.LENGTH_LONG).show()
                    requestNotificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
                else -> {
                    requestNotificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close() // Release TFLite resources
        Log.i("MainActivity", "TFLite Model Closed.")
    }
}
