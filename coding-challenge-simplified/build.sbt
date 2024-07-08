// Here's a quick glimpse of what a multi-project build looks like for this
// build, with only one "subproject" defined, called `root`:
val scala3Version = "3.4.2"

lazy val root = (project in file(".")).settings(
  inThisBuild(
    List(
      name := "scala-djl-project",
      version := "0.1.0-SNAPSHOT",
      scalaVersion := scala3Version,
      libraryDependencies ++= Seq(
        "org.scala-lang.modules" %% "scala-parser-combinators" % "2.3.0",
        "org.slf4j" % "slf4j-simple" % "1.7.36",
        "ai.djl" % "api" % "0.25.0",
        "ai.djl.mxnet" % "mxnet-engine" % "0.25.0"
      )
    )
  ),
  // logLevel := Level.Debug,
  name := "hello-world"
)
