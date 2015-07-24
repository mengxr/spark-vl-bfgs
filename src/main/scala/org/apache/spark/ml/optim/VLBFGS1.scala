package org.apache.spark.ml.optim

import java.util.Random

import org.apache.spark.{SparkContext, SparkConf}

import scala.collection.mutable
import scala.language.implicitConversions

import com.github.fommil.netlib.BLAS.{getInstance => blas}

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
    useDisk = true, useMemory = true, deserialized = true, replication = 8)

  private type RDDVector = RDD[Vector]
  private type Inner = mutable.Map[(Int, Int), Double]

  private def newInner: Inner = mutable.Map.empty.withDefaultValue(0.0)

  /** number of corrections */
  private val m: Int = 10

  /** max number of iterations */
  private val maxIter: Int = 20

  /** step size */
  private val stepSize: Double = 0.5

  /**
   * Runs vector-free L-BFGS and return the solution as an RDD[Vector]. This is different from the
   * paper, in which s_i and y_i are cached. We cache x_i and g_i instead to avoid creating more
   * RDDs. The algorithm should be exactly the same as L-BFGS subject to numeric errors.
   */
  def solve(data: RDD[Array[LabeledPoint]]): RDDVector = {
    require(data.getStorageLevel != StorageLevel.NONE)
    val sc = data.context
    val xx: Array[RDDVector] = Array.fill(maxIter)(null)
    val gg: Array[RDDVector] = Array.fill(maxIter)(null)
    val XX: Inner = newInner
    val XG: Inner = newInner
    val GG: Inner = newInner
    var x: RDDVector = init(data).setName("x0").persist(storageLevel)
    for (k <- 0 until maxIter) {
      // println(s"x($k) = ${x.first()}")
      // TODO: clean old vectors
      xx(k) = x

      // compute gradient
      val g = gradient(data, x).setName(s"g$k").persist(storageLevel)
      // println(s"g($k) = ${g.first()}")

      gg(k) = g

      // update XX, XG, and GG
      val start = math.max(k - m, 0)
      val tasks = (
        (start to k).map(i => ("xx", i)) ++ // update XX
        (start to k).map(j => ("xg_x", j)) ++ // update XG (x side)
        (start until k).map(i => ("xg_g", i)) ++ // update XG (g side)
        (start to k).map(j => ("gg", j)) // update GG
      ).toParArray
      tasks.foreach {
        case ("xx", i) =>
          val d = dot(x, xx(i))
          XX((i, k)) = d
          XX((k, i)) = d
        case ("xg_x", j) =>
          XG((k, j)) = dot(x, gg(j))
        case ("xg_g", i) =>
          XG((i, k)) = dot(xx(i), g)
        case ("gg", j) =>
         val d = dot(g, gg(j))
         GG((j, k)) = d
         GG((k, j)) = d
      }

      // println(s"XX: $XX")
      // println(s"XG: $XG")
      // println(s"GG: $GG")

      // compute p
      val (theta, tau) = computeDirection(k, XX, XG, GG)
      val scaled = (lastM(xx, k).zip(theta) ++ lastM(gg, k).zip(tau))
        .flatMap { case (db, d) =>
          Option(db).map {_.map { v =>
            val output = Vectors.zeros(v.size)
            BLAS.axpy(d, v, output)
            output
          }}
        }
      val p = new UnionRDD(sc, scaled.toSeq).treeSum()
      // println(s"p($k) = ${p.first()}")

      // TODO: line search

      x = axpy(stepSize, p, x).setName(s"x${k + 1}").persist(storageLevel)

      // clean old ones
      if (k > m) {
        xx(k - m - 1).unpersist(false)
        gg(k - m - 1).unpersist(false)
      }
    }

    x
  }

  private def lastM(xx: Array[RDDVector], k: Int): Array[RDDVector] = {
    (k - m to k).map { i =>
      if (i < 0) null else xx(i)
    }.toArray
  }

  private def computeDirection(k: Int, XX: Inner, XG: Inner, GG: Inner): (Array[Double], Array[Double]) = {
    val theta = new Array[Double](m + 1)
    val tau = new Array[Double](m + 1)
    val alpha = new Array[Double](m)
    tau(m) = -1.0
    if (k == 0) {
      return (theta, tau)
    }
    val start = math.max(k - m, 0)
    for (i <- (start until k).reverse) {
      val j = i - (k - m)
      var sum = 0.0
      for (l <- k - m to k) {
        sum += (XX((i + 1, l)) - XX((i, l))) * theta(l - (k - m)) +
          (XG((i + 1, l)) - XG((i, l))) * tau(l - (k - m))
      }
      val a = sum / (XG((i + 1, i + 1)) - XG((i + 1, i)) - XG((i, i + 1)) + XG((i, i)))
      assert(!a.isNaN, s"failed at iteration $k with i=$i.")
      alpha(j) = a
      tau(j + 1) += -a
      tau(j) += a
    }
    val scal = (XG((k, k)) - XG((k, k - 1)) - XG((k - 1, k)) + XG((k - 1, k - 1))) /
      (GG((k, k)) - 2.0 * GG((k, k - 1)) + GG((k - 1, k - 1)))
    blas.dscal(m + 1, scal, theta, 1)
    blas.dscal(m + 1, scal, tau, 1)
    for (i <- start until k) {
      val j = i - (k - m)
      var sum = 0.0
      for (l <- k - m to k) {
        sum += (XG((l, i + 1)) - XG((l, i))) * theta(l - (k - m)) +
          (GG((l, i + 1)) - GG(l, i)) * tau(l - (k - m))
      }
      val b = alpha(j) - sum / (XG((i + 1, i + 1)) - XG((i + 1, i)) - XG((i, i + 1)) + XG((i, i)))
      assert(!b.isNaN)
      theta(j + 1) += b
      theta(j) += -b
    }
    (theta, tau)
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

  private def dot(dx: RDDVector, dy: RDDVector): Double = {
    if (dx.eq(dy)) {
      dx.map { x => BLAS.dot(x, x) }.sum()
    } else {
      dx.zip(dy).map { case (x, y) => BLAS.dot(x, y) }.sum()
    }
  }

  private def axpy(a: Double, dx: RDDVector, dy: RDDVector): RDDVector = {
    dx.zip(dy).map { case (x, y) =>
      val out = y.copy
      BLAS.axpy(a, x, out)
      out
    }
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("VLBFGS").setMaster("local[*]")
    val sc = new SparkContext(conf)
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

    // println(s"x_exact = $xExact")
    // println(s"x_vlbfgs = $x")

    sc.stop()
  }
}
