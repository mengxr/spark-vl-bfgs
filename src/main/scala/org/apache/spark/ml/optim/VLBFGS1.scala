package org.apache.spark.ml.optim

import java.util.Random

import scala.language.implicitConversions

import org.apache.hadoop.fs.{FileSystem, Path}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.optim.VectorFreeLBFGS.{Oracle, VectorSpace}
import org.apache.spark.ml.optim.VectorRDDFunctions._
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.{RDD, UnionRDD}
import org.apache.spark.storage.StorageLevel

/**
 * Vector-free L-BFGS.
 *
 * First version, use RDD[Vector] with a single record to store model, and an array of RDD[Vector]
 * to store corrections. So driver doesn't store anything large.
 */
object VLBFGS1 {

  private val storageLevel = StorageLevel(
    useDisk = true, useMemory = true, deserialized = true, replication = 3)

  private type RDDVector = RDD[Vector]

  /** number of corrections */
  private val m: Int = 10

  /** max number of iterations */
  var maxIter: Int = 20

  /** step size */
  var stepSize: Double = 0.5

  val vs: VectorSpace[RDDVector] = new VectorSpace[RDDVector] {

    override def clean(v: RDDVector): Unit = {
      v.unpersist(false)
      val conf = v.context.hadoopConfiguration
      val fs = FileSystem.get(conf)
      v.getCheckpointFile.foreach { file =>
        fs.delete(new Path(file), true)
      }
    }

    override def combine(vv: (Double, RDDVector)*): RDDVector = {
      val sc = vv.head._2.context
      val scaled = vv.map { case (a, dv) =>
        dv.map { v =>
          val output = Vectors.zeros(v.size)
          BLAS.axpy(a, v, output)
          output
        }
      }
      new UnionRDD(sc, scaled).treeSum()
    }

    override def dot(v1: RDDVector, v2: RDDVector): Double = {
      if (v1.eq(v2)) {
        v1.map { x => BLAS.dot(x, x) }.sum()
      } else {
        v1.zip(v2).map { case (x1, x2) => BLAS.dot(x1, x2) }.sum()
      }
    }
  }

  /**
   * Runs vector-free L-BFGS and return the solution as an RDD[Vector]. This is different from the
   * paper, in which s_i and y_i are cached. We cache x_i and g_i instead to avoid creating more
   * RDDs. The algorithm should be exactly the same as L-BFGS subject to numeric errors.
   */
  def solve(data: RDD[Array[LabeledPoint]]): RDDVector = {
    require(data.getStorageLevel != StorageLevel.NONE)

    val oracle = new Oracle[RDDVector](m, vs)

    var x: RDDVector = init(data).setName("x0").persist(storageLevel)
    x.checkpoint()
    for (k <- 0 until maxIter) {
      val g = gradient(data, x).setName(s"g$k").persist(storageLevel)
      g.checkpoint()
      val gn = math.sqrt(vs.dot(g, g))
      println(s"norm(g($k)) = $gn")
      val p = oracle.findDirection(x, g)
      // TODO: line search
      x = vs.combine((stepSize, p), (1.0, x)).setName(s"x${k + 1}").persist(storageLevel)
      x.checkpoint()
    }

    x
  }

  private def init(data: RDD[Array[LabeledPoint]]): RDD[Vector] = {
    val sc = data.context
    val size = data.map(_.head.features.size).first()
    sc.parallelize(Seq(0), 1).map(_ => Vectors.zeros(size))
  }

  /**
   * Computes least squares gradient.
   * @param data data points
   * @param dx current weights
   * @return gradient vector stored in an RDD[Vector]
   */
  private def gradient(data: RDD[Array[LabeledPoint]], dx: RDD[Vector]): RDD[Vector] = {
    data.cartesian(dx).map { case (points, x) =>
      val g = Vectors.zeros(x.size)
      points.foreach { case LabeledPoint(b, a) =>
        val err = BLAS.dot(a, x) - b
        BLAS.axpy(err, a, g)
      }
      g
    }.treeSum()
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("VLBFGS").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("/tmp/checkpoint")
    val n = 1000
    val p = 100
    val random = new Random(0L)
    val xExact = Vectors.dense(Array.fill(p)(random.nextDouble()))
    val data = RandomRDDs.normalVectorRDD(sc, n, p, 4, 11L).mapPartitionsWithIndex { (idx, part) =>
      val random = new Random(100 + idx)
      part.map { v =>
        val target = BLAS.dot(v, xExact) + 0.1 * random.nextGaussian()
        LabeledPoint(target, v)
      }
    }.glom()
    .cache()

    val x = solve(data).first()

    println(s"x_exact = $xExact")
    println(s"x_vlbfgs = $x")

    sc.stop()
  }
}
