# ONNX Runtime and MediaPipe both use JNI + reflection, so their entry points
# must survive shrinking. Only relevant once isMinifyEnabled is turned on.
-keep class ai.onnxruntime.** { *; }
-keep class com.google.mediapipe.** { *; }
-keep class org.opencv.** { *; }
