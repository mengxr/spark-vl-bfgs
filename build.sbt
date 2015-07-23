net.virtualvoid.sbt.graph.Plugin.graphSettings

organization := "com.github.mengxr"

name := "spark-vlbfgs"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.4.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.1.5" % Test
