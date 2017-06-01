import numpy

from radiomics import base, imageoperations


class RadiomicsFirstOrder(base.RadiomicsFeaturesBase):
  r"""
  First-order statistics describe the distribution of voxel intensities within the image region defined by the mask
  through commonly used and basic metrics.

  Let:

  - :math:`\textbf{X}` be a set of :math:`N` voxels included in the ROI
  - :math:`\textbf{P}(i)` be the first order histogram with :math:`N_l` discrete intensity levels,
    where :math:`N_l` is the number of non-zero bins, equally spaced from 0 with a width defined in the ``binWidth``
    parameter.
  - :math:`p(i)` be the normalized first order histogram and equal to :math:`\frac{\textbf{P}(i)}{\sum{\textbf{P}(i)}}`

  Following additional settings are possible:

  - voxelArrayShift [0]: Integer, This amount is added to the gray level intensity in features Energy, Total Energy and
    RMS, this is to prevent negative values. *If using CT data, or data normalized with mean 0, consider setting this
    parameter to a fixed value (e.g. 2000) that ensures non-negative numbers in the image. Bear in mind however, that
    the larger the value, the larger the volume confounding effect will be.*
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    super(RadiomicsFirstOrder, self).__init__(inputImage, inputMask, **kwargs)

    self.pixelSpacing = inputImage.GetSpacing()
    self.voxelArrayShift = kwargs.get('voxelArrayShift', 0)

    self._initLesionWiseCalculation()

  def _initLesionWiseCalculation(self):
    super(RadiomicsFirstOrder, self)._initLesionWiseCalculation()
    self.targetVoxelArray = self.imageArray[self.ROICoordinates].astype('float')

    self.logger.debug('Feature class initialized')

  def _moment(self, a, moment=1, axis=0):
    r"""
    Calculate n-order moment of an array for a given axis
    """

    if moment == 1:
      return numpy.float64(0.0)
    else:
      mn = numpy.mean(a, axis, keepdims=True)
      s = numpy.power((a - mn), moment)
      return numpy.mean(s, axis)

  def getEnergyFeatureValue(self):
    r"""
    **1. Energy**

    .. math::

      \textit{energy} = \displaystyle\sum^{N}_{i=1}{(\textbf{X}(i) + c)^2}

    Here, :math:`c` is optional value, defined by ``voxelArrayShift``, which shifts the intensities to prevent negative
    values in :math:`\textbf{X}`. This ensures that voxels with the lowest gray values contribute the least to Energy,
    instead of voxels with gray level intensity closest to 0.

    Energy is a measure of the magnitude of voxel values in an image. A larger values implies a greater sum of the
    squares of these values.

    .. note::

      This feature is volume-confounded, a larger value of :math:`c` increases the effect of volume-confounding.
    """

    shiftedParameterArray = self.targetVoxelArray + self.voxelArrayShift

    return numpy.sum(shiftedParameterArray ** 2)

  def getTotalEnergyFeatureValue(self):
    r"""
    **2. Total Energy**

    .. math::

      \textit{total energy} = V_{voxel}\displaystyle\sum^{N}_{i=1}{(\textbf{X}(i) + c)^2}

    Here, :math:`c` is optional value, defined by ``voxelArrayShift``, which shifts the intensities to prevent negative
    values in :math:`\textbf{X}`. This ensures that voxels with the lowest gray values contribute the least to Energy,
    instead of voxels with gray level intensity closest to 0.

    Total Energy is the value of Energy feature scaled by the volume of the voxel in cubic mm.

    .. note::

      This feature is volume-confounded, a larger value of :math:`c` increases the effect of volume-confounding.
    """

    x, y, z = self.pixelSpacing
    cubicMMPerVoxel = x * y * z

    return cubicMMPerVoxel * self.getEnergyFeatureValue()

  def getEntropyFeatureValue(self):
    r"""
    **3. Entropy**

    .. math::

      \textit{entropy} = -\displaystyle\sum^{N_l}_{i=1}{p(i)\log_2\big(p(i)+\epsilon\big)}

    Here, :math:`\epsilon` is an arbitrarily small positive number (:math:`\approx 2.2\times10^{-16}`).

    Entropy specifies the uncertainty/randomness in the image values. It measures the average amount of information
    required to encode the image values.
    """

    eps = numpy.spacing(1)
    binEdges = imageoperations.getBinEdges(self.binWidth, self.targetVoxelArray)
    bins = numpy.histogram(self.targetVoxelArray, binEdges)[0]

    sumBins = bins.sum()
    if sumBins == 0:  # No segmented voxels
      return 0

    bins = bins + eps
    bins = bins / float(sumBins)
    return -1.0 * numpy.sum(bins * numpy.log2(bins))

  def getMinimumFeatureValue(self):
    r"""
    **4. Minimum**

    .. math::

      \textit{minimum} = \min(\textbf{X})
    """

    return numpy.min(self.targetVoxelArray)

  def get10PercentileFeatureValue(self):
    r"""
    **5. 10th percentile**

    The 10\ :sup:`th` percentile of :math:`\textbf{X}`
    """

    return numpy.percentile(self.targetVoxelArray, 10)

  def get90PercentileFeatureValue(self):
    r"""
    **6. 90th percentile**

    The 90\ :sup:`th` percentile of :math:`\textbf{X}`
    """

    return numpy.percentile(self.targetVoxelArray, 90)

  def getMaximumFeatureValue(self):
    r"""
    **7. Maximum**

    .. math::

      \textit{maximum} = \max(\textbf{X})

    The maximum gray level intensity within the ROI.
    """

    return numpy.max(self.targetVoxelArray)

  def getMeanFeatureValue(self):
    r"""
    **8. Mean**

    .. math::

      \textit{mean} = \frac{1}{N}\displaystyle\sum^{N}_{i=1}{\textbf{X}(i)}

    The average gray level intensity within the ROI.
    """

    return numpy.mean(self.targetVoxelArray)

  def getMedianFeatureValue(self):
    r"""
    **9. Median**

    The median gray level intensity within the ROI.
    """

    return numpy.median(self.targetVoxelArray)

  def getInterquartileRangeFeatureValue(self):
    r"""
    **10. Interquartile Range**

    .. math::

      \textit{interquartile range} = \textbf{P}_{75} - \textbf{P}_{25}

    Here :math:`\textbf{P}_{25}` and :math:`\textbf{P}_{75}` are the 25\ :sup:`th` and 75\ :sup:`th` percentile of the
    image array, respectively.
    """

    return numpy.percentile(self.targetVoxelArray, 75) - numpy.percentile(self.targetVoxelArray, 25)

  def getRangeFeatureValue(self):
    r"""
    **11. Range**

    .. math::

      \textit{range} = \max(\textbf{X}) - \min(\textbf{X})

    The range of gray values in the ROI.
    """

    return numpy.max(self.targetVoxelArray) - numpy.min(self.targetVoxelArray)

  def getMeanAbsoluteDeviationFeatureValue(self):
    r"""
    **12. Mean Absolute Deviation (MAD)**

    .. math::

      \textit{MAD} = \frac{1}{N}\displaystyle\sum^{N}_{i=1}{|\textbf{X}(i)-\bar{X}|}

    Mean Absolute Deviation is the mean distance of all intensity values from the Mean Value of the image array.
    """

    return numpy.mean(numpy.absolute((numpy.mean(self.targetVoxelArray) - self.targetVoxelArray)))

  def getRobustMeanAbsoluteDeviationFeatureValue(self):
    r"""
    **13. Robust Mean Absolute Deviation (rMAD)**

    .. math::

      \textit{rMAD} = \frac{1}{N_{10-90}}\displaystyle\sum^{N_{10-90}}_{i=1}
      {|\textbf{X}_{10-90}(i)-\bar{X}_{10-90}|}

    Robust Mean Absolute Deviation is the mean distance of all intensity values
    from the Mean Value calculated on the subset of image array with gray levels in between, or equal
    to the 10\ :sup:`th` and 90\ :sup:`th` percentile.
    """

    prcnt10 = self.get10PercentileFeatureValue()
    prcnt90 = self.get90PercentileFeatureValue()
    percentileArray = self.targetVoxelArray[(self.targetVoxelArray >= prcnt10) * (self.targetVoxelArray <= prcnt90)]

    return numpy.mean(numpy.absolute(percentileArray - numpy.mean(percentileArray)))

  def getRootMeanSquaredFeatureValue(self):
    r"""
    **14. Root Mean Squared (RMS)**

    .. math::

      \textit{RMS} = \sqrt{\frac{1}{N}\sum^{N}_{i=1}{(\textbf{X}(i) + c)^2}}

    Here, :math:`c` is optional value, defined by ``voxelArrayShift``, which shifts the intensities to prevent negative
    values in :math:`\textbf{X}`. This ensures that voxels with the lowest gray values contribute the least to RMS,
    instead of voxels with gray level intensity closest to 0.

    RMS is the square-root of the mean of all the squared intensity values. It is another measure of the magnitude of
    the image values. This feature is volume-confounded, a larger value of :math:`c` increases the effect of
    volume-confounding.
    """

    # If no voxels are segmented, prevent division by 0 and return 0
    if self.targetVoxelArray.size == 0:
      return 0

    shiftedParameterArray = self.targetVoxelArray + self.voxelArrayShift
    return numpy.sqrt((numpy.sum(shiftedParameterArray ** 2)) / float(shiftedParameterArray.size))

  def getStandardDeviationFeatureValue(self):
    r"""
    **15. Standard Deviation**

    .. math::

      \textit{standard deviation} = \sqrt{\frac{1}{N}\sum^{N}_{i=1}{(\textbf{X}(i)-\bar{X})^2}}

    Standard Deviation measures the amount of variation or dispersion from the Mean Value. By definition,
    :math:\textit{standard deviation} = \sqrt{\textit{variance}}
    """

    return numpy.std(self.targetVoxelArray)

  def getSkewnessFeatureValue(self, axis=0):
    r"""
    **16. Skewness**

    .. math::

      \textit{skewness} = \displaystyle\frac{\mu_3}{\sigma^3} =
      \frac{\frac{1}{N}\sum^{N}_{i=1}{(\textbf{X}(i)-\bar{X})^3}}
      {\left(\sqrt{\frac{1}{N}\sum^{N}_{i=1}{(\textbf{X}(i)-\bar{X})^2}}\right)^3}

    Where :math:`\mu_3` is the 3\ :sup:`rd` central moment.

    Skewness measures the asymmetry of the distribution of values about the Mean value. Depending on where the tail is
    elongated and the mass of the distribution is concentrated, this value can be positive or negative.

    Related links:

    https://en.wikipedia.org/wiki/Skewness

    .. note::

      In case of a flat region, the standard deviation and 4\ :sup:`rd` central moment will be both 0. In this case, a
      value of 0 is returned.
    """

    m2 = self._moment(self.targetVoxelArray, 2, axis)
    m3 = self._moment(self.targetVoxelArray, 3, axis)

    if m2 == 0:  # Flat Region
      return 0

    return m3 / m2 ** 1.5

  def getKurtosisFeatureValue(self, axis=0):
    r"""
    **17. Kurtosis**

    .. math::

      \textit{kurtosis} = \displaystyle\frac{\mu_4}{\sigma^4} =
      \frac{\frac{1}{N}\sum^{N}_{i=1}{(\textbf{X}(i)-\bar{X})^4}}
      {\left(\frac{1}{N}\sum^{N}_{i=1}{(\textbf{X}(i)-\bar{X}})^2\right)^2}

    Where :math:`\mu_4` is the 4\ :sup:`th` central moment.

    Kurtosis is a measure of the 'peakedness' of the distribution of values in the image ROI. A higher kurtosis implies
    that the mass of the distribution is concentrated towards the tail(s) rather than towards the mean. A lower kurtosis
    implies the reverse: that the mass of the distribution is concentrated towards a spike near the Mean value.

    Related links:

    https://en.wikipedia.org/wiki/Kurtosis

    .. note::

      In case of a flat region, the standard deviation and 4\ :sup:`rd` central moment will be both 0. In this case, a
      value of 0 is returned.
    """

    m2 = self._moment(self.targetVoxelArray, 2, axis)
    m4 = self._moment(self.targetVoxelArray, 4, axis)

    if m2 == 0:  # Flat Region
      return 0

    return m4 / m2 ** 2.0

  def getVarianceFeatureValue(self):
    r"""
    **18. Variance**

    .. math::

      \textit{variance} = \frac{1}{N}\displaystyle\sum^{N}_{i=1}{(\textbf{X}(i)-\bar{X})^2}

    Variance is the the mean of the squared distances of each intensity value from the Mean value. This is a measure of
    the spread of the distribution about the mean. By definition, :math:`\textit{variance} = \sigma^2`
    """

    return numpy.std(self.targetVoxelArray) ** 2

  def getUniformityFeatureValue(self):
    r"""
    **19. Uniformity**

    .. math::

      \textit{uniformity} = \displaystyle\sum^{N_l}_{i=1}{p(i)^2}

    Uniformity is a measure of the sum of the squares of each intensity value. This is a measure of the heterogeneity of
    the image array, where a greater uniformity implies a greater heterogeneity or a greater range of discrete intensity
    values.
    """

    eps = numpy.spacing(1)
    binEdges = imageoperations.getBinEdges(self.binWidth, self.targetVoxelArray)
    bins = numpy.histogram(self.targetVoxelArray, binEdges)[0]
    sumBins = bins.sum()

    if sumBins == 0:  # No segmented voxels
      return 0

    bins = bins / (float(sumBins + eps))
    return numpy.sum(bins ** 2)
