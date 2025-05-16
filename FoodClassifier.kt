package com.example.foodhealthclassifier

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

object FoodClassifier {

    private lateinit var model: Module

    fun loadModel(context: Context)
    {
        val fileName = "model_optimized.ptl"
        val file = File(context.cacheDir, fileName)

        if (!file.exists()) {
            context.assets.open(fileName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }

        model = Module.load(file.absolutePath)

    }


    fun classify(bitmap: Bitmap): String {
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            floatArrayOf(0f, 0f, 0f),       // mean
            floatArrayOf(255f, 255f, 255f)  // std
        )

        val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
        val output = outputTensor.dataAsFloatArray

        // Get max index and confidence
        val maxIndex = output.indices.maxByOrNull { output[it] } ?: -1
        val confidence = output[maxIndex] * 100

        val label = when (maxIndex) {
            0 -> "Healthy"
            1 -> "Unhealthy"
            else -> "Unknown"
        }

        return "$label (${String.format("%.1f", confidence)}%)"
    }
}

private fun ERROR.exists() {
    TODO("Not yet implemented")
}

annotation class ERROR
