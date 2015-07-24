package org.apache.spark.ml.optim

import scala.reflect.ClassTag

import breeze.linalg.{DenseMatrix => BDM}
import com.github.fommil.netlib.BLAS.{getInstance => blas}

object VectorFreeLBFGS {

  // private type VEC = RDD[(Int, Vector)]
  // private type MAT = RDD[((Int, Int), Matrix)]

  trait VectorSpace[V] {
    /** inner product */
    def dot(v1: V, v2: V): Double

    /** linear combination */
    def combine(vv: (Double, V)*): V

    def clean(v: V): Unit
  }

  class Oracle[V: ClassTag](val m: Int, val vs: VectorSpace[V]) {

    require(m > 0)

    private val m1: Int = m + 1
    private var k: Int = 0
    private val xx: Array[V] = new Array[V](m1)
    private val gg: Array[V] = new Array[V](m1)
    private val XX: BDM[Double] = BDM.zeros(m1, m1)
    private val XG: BDM[Double] = BDM.zeros(m1, m1)
    private val GG: BDM[Double] = BDM.zeros(m1, m1)

    def findDirection(x: V, g: V): V = {
      shift(xx, x)
      shift(gg, g)
      updateInnerProducts()
      val p = computeDirection()
      k += 1
      p
    }

    private def updateInnerProducts(): Unit = {
      shift(XX)
      shift(XG)
      shift(GG)
      val start = math.max(m - k, 0)
      ((start to m).map(i => ("XX", i, m)) ++
        (start to m).map(i => ("XG", i, m)) ++
        (start until m).map(i => ("XG", m, i)) ++
        (start to m).map(i => ("GG", i, m)))
        .par.foreach(updateInnerProduct)
    }

    private def updateInnerProduct(task: (String, Int, Int)): Unit = task match {
      case ("XX", i, j) =>
        val d = vs.dot(xx(i), xx(j))
        XX(i, j) = d
        XX(j, i) = d
      case ("XG", i, j) =>
        XG(i, j) = vs.dot(xx(i), gg(j))
      case ("GG", i, j) =>
        val d = vs.dot(gg(i), gg(j))
        GG(i, j) = d
        GG(j, i) = d
    }

    private def shift(vv: Array[V], v: V): Unit = {
      Option(vv.head).foreach(vs.clean)
      for (i <- 0 until m) {
        vv(i) = vv(i + 1)
      }
      vv(m) = v
    }

    private def shift(VV: BDM[Double]): Unit = {
      for (i <- 0 until m; j <- 0 until m) {
        VV(i, j) = VV(i + 1, j + 1)
      }
    }

    def computeDirection(): V = {
      println(s"XX =\n$XX")
      println(s"XG =\n$XG")
      println(s"GG =\n$GG")
      if (k == 0) {
        return vs.combine((-1.0, gg(m)))
      }
      val theta = new Array[Double](m1)
      val tau = new Array[Double](m1)
      tau(m) = -1.0
      val alpha = new Array[Double](m)
      val start = math.max(m - k, 0)
      for (i <- (start until m).reverse) {
        val i1 = i + 1
        var sum = 0.0
        for (j <- 0 to m) {
          sum += (XX(i1, j) - XX(i, j)) * theta(j) + (XG(i1, j) - XG(i, j)) * tau(j)
        }
        assert(sum != 0.0)
        val a = sum / (XG(i1, i1) - XG(i1, i) - XG(i, i1) + XG(i, i))
        assert(!a.isNaN)
        alpha(i) = a
        tau(i + 1) += -a
        tau(i) += a
      }
      val mm1 = m - 1
      val scal = (XG(m, m) - XG(m, mm1) - XG(mm1, m) + XG(mm1, mm1)) /
        (GG(m, m) - 2.0 * GG(m, mm1) + GG(mm1, mm1))
      blas.dscal(m1, scal, theta, 1)
      blas.dscal(m1, scal, tau, 1)
      for (i <- start until m) {
        val i1 = i + 1
        var sum = 0.0
        for (j <- 0 to m) {
          sum += (XG(j, i1) - XG(j, i)) * theta(j) + (GG(j, i1) - GG(j, i)) * tau(j)
        }
        val b = alpha(i) - sum / (XG(i1, i1) - XG(i1, i) - XG(i, i1) + XG(i, i))
        assert(!b.isNaN)
        theta(i + 1) += b
        theta(i) += -b
      }
      vs.combine((theta.zip(xx) ++ tau.zip(gg)).filter(_._1 != 0.0): _*)
    }
  }
}
