package org.apache.spark.ml.optim

import scala.language.implicitConversions

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._

class VectorRDDFunctions(self: RDD[Vector]) {

  def treeSum(depth: Int = 2): RDD[Vector] = {
    val zeroValue: Vector = null
    val seqOp = (s: Vector, v: Vector) => {
      if (s != null) {
        BLAS.axpy(1.0, v, s)
        s
      } else {
        v.copy.toDense
      }
    }
    val combOp = (s1: Vector, s2: Vector) => {
      // TODO: handle empty partitions
      BLAS.axpy(1.0, s2, s1)
      s1
    }
    require(depth >= 1, s"Depth must be greater than or equal to 1 but got $depth.")
    val aggregatePartition = (it: Iterator[Vector]) => it.aggregate(zeroValue)(seqOp, combOp)
    var partiallyAggregated = self.mapPartitions(it => Iterator(aggregatePartition(it)))
    var numPartitions = partiallyAggregated.partitions.length
    val scale = math.max(math.pow(numPartitions, 1.0 / depth), 2.0)
    while (numPartitions > 1) {
      numPartitions = math.ceil(numPartitions / scale).toInt
      val curNumPartitions = numPartitions
      partiallyAggregated = partiallyAggregated.mapPartitionsWithIndex {
        (i, iter) => iter.map((i % curNumPartitions, _))
      }.reduceByKey(new HashPartitioner(curNumPartitions), combOp)
      .values
    }
    require(partiallyAggregated.partitions.length == 1)
    partiallyAggregated
  }
}

object VectorRDDFunctions {
  implicit def fromVectorRDD(rdd: RDD[Vector]): VectorRDDFunctions = new VectorRDDFunctions(rdd)
}
