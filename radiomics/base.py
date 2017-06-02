import inspect
import logging
import traceback

import numpy
import SimpleITK as sitk
import six

import radiomics

class RadiomicsFeaturesBase(object):
  """
  This is the abstract class, which defines the common interface for the feature classes. All feature classes inherit
  (directly of indirectly) from this class.

  At initialization, image and labelmap are passed as SimpleITK image objects (``inputImage`` and ``inputMask``,
  respectively.) The motivation for using SimpleITK images as input is to keep the possibility of reusing the
  optimized feature calculators implemented in SimpleITK in the future. If either the image or the mask is None,
  initialization fails and a warning is logged (does not raise an error).

  Logging is set up using a child logger from the parent 'radiomics' logger. This retains the toolbox structure in
  the generated log.

  The following variables are instantiated at initialization:

  - binWidth: bin width, as specified in ``**kwargs``. If key is not present, a default value of 25 is used.
  - label: label value of Region of Interest (ROI) in labelmap. If key is not present, a default value of 1 is used.
  - verbose: boolean indication whether or not to provide progress reporting to the output.
  - featureNames: list containing the names of features defined in the feature class. See :py:func:`getFeatureNames`
  - inputImage: Simple ITK image object of the input image
  - inputMask: Simple ITK image object of the input labelmap
  - imageArray: numpy array of the gray values in the input image
  - maskArray: numpy array with elements set to 1 where labelmap = label, 0 otherwise
  - matrix: numpy array of the gray values in the input image (with gray values inside ROI discretized when necessary in
    the texture feature classes).
  - matrixCoordinates: tuple of 3 numpy arrays containing the z, x and y coordinates of the voxels included in the ROI,
    respectively. Length of each array is equal to total number of voxels inside ROI.
  - targetVoxelArray: flattened numpy array of gray values inside ROI.
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    self.logger = logging.getLogger(self.__module__)
    self.logger.debug('Initializing feature class')
    # check if the handler to stderr is set and its level is INFO or lower
    if logging.NOTSET < radiomics.handler.level <= logging.INFO:
      self.progressReporter = radiomics._getProgressReporter
    else:
      self.progressReporter = radiomics._DummyProgressReporter

    self.kwargs = kwargs
    self.binWidth = kwargs.get('binWidth', 25)
    self.label = kwargs.get('label', 1)
    self.voxelWise = kwargs.get('voxelWise', False)

    self.coefficients = {}

    # all features are disabled by default
    self.disableAllFeatures()

    self.featureNames = self.getFeatureNames()

    self.inputImage = inputImage
    self.inputMask = inputMask

    if inputImage is None or inputMask is None:
      self.logger.warning('Missing input image or mask')
      return

  def _initLesionWiseCalculation(self):
    self.imageArray = sitk.GetArrayFromImage(self.inputImage)
    self.maskArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)

    self.maskCoordinates = numpy.where(self.maskArray != 0)
    self.size = numpy.max(self.maskCoordinates, 1) - numpy.min(self.maskCoordinates, 1) + 1

  def _initVoxelWiseCalculation(self):
    kernelDistance = self.kwargs.get('kernelDistance', 1)
    masked = self.kwargs.get('maskedKernel', True)

    self.imageArray = sitk.GetArrayFromImage(self.inputImage)
    self.ROIArray = (sitk.GetArrayFromImage(self.inputMask) == self.label)

    # Mask Coordinates and kernel coordinates define voxels that MAY be assessed (depending on the type of border
    # handling, this is either all voxels in the ROI (masked kernel) or all voxels in the image (full kernel)
    # Mask coordinates will change for each calculated voxel, but kernel Coordinates will remain constant
    if masked:  # maskArray must be initialized depending on how the borders are handled (masked or not)
      self.maskArray = self.ROIArray.copy()  # will exclude voxels from outside the ROI
      maskCoordinates = numpy.where(self.maskArray != 0)
      self.ROICoordinates = set(zip(*maskCoordinates))  # Voxels in the ROI, i.e. voxels to process
    else:
      self.maskArray = numpy.ones(self.ROIArray.shape, dtype='bool')  # include all voxels inside kernel
      maskCoordinates = numpy.where(self.maskArray != 0)
      self.ROICoordinates = set(zip(*numpy.where(self.ROIArray != 0)))  # Voxels in the ROI, i.e. voxels to process

    self.kernelCoordinates = set(zip(*maskCoordinates))
    self.size = numpy.max(maskCoordinates, 1) - numpy.min(maskCoordinates, 1) + 1

    # Generate Kernel
    distances = six.moves.range(1, kernelDistance + 1)
    self.kernel = radiomics.imageoperations.generateAngles(self.size,
                                                           distances=distances,
                                                           force2Dextraction=self.kwargs.get('force2D', False),
                                                           force2Ddimension=self.kwargs.get('force2Ddimension', 0))
    self.kernel = numpy.concatenate(([(0, 0, 0)], self.kernel, self.kernel * -1))
    self.size = numpy.max(self.kernel, 0) - numpy.min(self.kernel, 0) + 1

  def _initCalculation(self):
    pass

  def _applyBinning(self):
      self.matrix, self.binEdges = radiomics.imageoperations.binImage(self.binWidth,
                                                                      self.imageArray,
                                                                      self.maskArray)
      self.coefficients['Ng'] = int(numpy.max(self.matrix[self.maskArray]))  # max gray level in the ROI
      self.coefficients['grayLevels'] = numpy.unique(self.matrix[self.maskArray])

  def enableFeatureByName(self, featureName, enable=True):
    """
    Enables or disables feature specified by ``featureName``. If feature is not present in this class, a lookup error is
    raised. ``enable`` specifies whether to enable or disable the feature.
    """
    if featureName not in self.featureNames:
      raise LookupError('Feature not found: ' + featureName)
    self.enabledFeatures[featureName] = enable

  def enableAllFeatures(self):
    """
    Enables all features found in this class for calculation.
    """
    for featureName in self.featureNames:
      self.enableFeatureByName(featureName, True)

  def disableAllFeatures(self):
    """
    Disables all features. Additionally resets any calculated features.
    """
    self.enabledFeatures = {}
    self.featureValues = {}

  @classmethod
  def getFeatureNames(cls):
    """
    Dynamically enumerates features defined in the feature class. Features are identified by the
    ``get<Feature>FeatureValue`` signature, where <Feature> is the name of the feature (unique on the class level).

    Found features are returned as a list of the feature names (``[<Feature1>, <Feature2>, ...]``).

    This function is called at initialization, found features are stored in the ``featureNames`` variable.
    """
    attributes = inspect.getmembers(cls)
    features = [a[0][3:-12] for a in attributes if a[0].startswith('get') and a[0].endswith('FeatureValue')]
    return features

  def calculateFeatures(self):
    """
    Calculates all features enabled in  ``enabledFeatures``. A feature is enabled if it's key is present in this
    dictionary and it's value is True.

    Calculated values are stored in the ``featureValues`` dictionary, with feature name as key and the calculated
    feature value as value. If an exception is thrown during calculation, the error is logged, and the value is set to
    NaN.
    """
    self.logger.debug('Calculating features')
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        try:
          # Use getattr to get the feature calculation methods, then use '()' to evaluate those methods
          self.featureValues[feature] = getattr(self, 'get%sFeatureValue' % feature)()
        except Exception:
          self.featureValues[feature] = numpy.nan
          self.logger.error('FAILED: %s', traceback.format_exc())

  def calculateSingleVoxel(self, index):
    # index = (0, 0, 0)
    # Apply Kernel
    # temp = [tuple(idx) for idx in (self.kernel + index)]
    kernelCoordinates = set(tuple(idx) for idx in (self.kernel + index))
    self.maskCoordinates = zip(*(self.ROICoordinates.intersection(kernelCoordinates)))
    if len(self.maskCoordinates[0]) <= 1:
      return

    self.maskArray[:] = False
    self.maskArray[self.maskCoordinates] = True

    self._initCalculation()

    # Calculate features in voxel
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        try:
          # Use getattr to get the feature calculation methods, then use '()' to evaluate those methods
          self.featureValues[feature][index] = getattr(self, 'get%sFeatureValue' % feature)()
        except Exception:
          self.featureValues[feature][index] = numpy.nan
          self.logger.error('FAILED: %s', traceback.format_exc())

  def calculateVoxels(self):
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        self.featureValues[feature] = numpy.zeros(self.ROIArray.shape, dtype='float')

    with self.progressReporter(self.ROICoordinates, 'Calculating voxels') as bar:
      for vox_idx in bar:
        self.calculateSingleVoxel(vox_idx)

    # Convert results to simple ITK image objects
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        self.featureValues[feature] = sitk.GetImageFromArray(self.featureValues[feature])
        self.featureValues[feature].CopyInformation(self.inputImage)
