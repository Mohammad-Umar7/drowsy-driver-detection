pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()        // AndroidX, MediaPipe tasks-vision
        mavenCentral()  // ONNX Runtime, OpenCV
    }
}

rootProject.name = "DrowsyDriver"
include(":app")
