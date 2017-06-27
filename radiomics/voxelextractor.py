# -*- coding: utf-8 -*-
from __future__ import print_function

import collections
from itertools import chain
import logging
import os

import pykwalify.core
import SimpleITK as sitk
import six

import radiomics
from radiomics import generalinfo, getFeatureClasses, getInputImageTypes, imageoperations


class RadiomicsVoxelExtractor:
  r"""
  Wrapper class for voxel-wise calculation of radiomic features.
  At and after initialisation various settings can be used to customize the resultant extraction.
  This includes which classes and features to use, as well as what should be done in terms of preprocessing the image
  and what images (original and/or filtered) should be used as input.

  Then a call to :py:func:`execute` calculates the radiomic features specified by these settings for the passed image
  and labelmap combination. This function can be called repeatedly in a batch process to calculate the radiomic
  features for all image and labelmap combinations.

  At initialization, a parameters file can be provided containing all necessary settings. This is done by passing the
  location of the file as the single argument in the initialization call, without specifying it as a keyword argument.
  If such a file location is provided, any additional kwargs are ignored.
  Alternatively, at initialisation, custom settings can be provided as keyword arguments, with the setting name as key
  and its value as the argument value (e.g. ``binWidth=25``). For mor information on possible settings and
  customization, see :ref:`Customizing the Extraction <radiomics-customization-label>`.

  By default, no features are enabled.
  By default, only `Original` input image is enabled (No filter applied).
  """

  def __init__(self, *args, **kwargs):
    self.logger = logging.getLogger(__name__)

    self.featureClasses = getFeatureClasses()

    self.settings = {}
    self.inputImages = {}
    self.enabledFeatures = {}

    if len(args) == 1 and isinstance(args[0], six.string_types):
      self.logger.info("Loading parameter file")
      self.loadParams(args[0])
    else:
      # Set default settings and update with and changed settings contained in kwargs
      self.settings = self._getDefaultSettings()
      if len(kwargs) > 0:
        self.logger.info('Applying custom settings')
        self.settings.update(kwargs)
      else:
        self.logger.info('No customized settings, applying defaults')

      self.logger.debug("Settings: %s", self.settings)

      self.inputImages = {'Original': {}}

      self.enabledFeatures = {}

    self.settings['voxelWise'] = True
    self._setTolerance()

  @classmethod
  def _getDefaultSettings(cls):
    """
    Returns a dictionary containg the default settings specified in this class. These settings cover global toolbox
    settings, such as ``enableCExtensions``, as well as the image pre-processing settings (e.g. resampling). Feature
    class specific are defined in the respective feature classes and and not included here. Similarly, filter specific
    settings are defined in ``imageoperations.py`` and also not included here.
    """
    return {'minimumROIDimensions': 1,
            'minimumROISize': None,  # Skip testing the ROI size by default
            'normalize': False,
            'normalizeScale': 1,
            'removeOutliers': None,
            'resampledPixelSpacing': None,  # No resampling by default
            'interpolator': 'sitkBSpline',  # Alternative: sitk.sitkBSpline
            'padDistance': 5,
            'distances': [1],
            'force2D': False,
            'force2Ddimension': 0,
            'label': 1,
            'enableCExtensions': True,
            'kernelDistance': 1,
            'maskedKernel': True}

  def _setTolerance(self):
    self.geometryTolerance = self.settings.get('geometryTolerance')
    if self.geometryTolerance is not None:
      sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(self.geometryTolerance)
      sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(self.geometryTolerance)

  def loadParams(self, paramsFile):
    """
    Parse specified parameters file and use it to update settings, enabled feature(Classes) and input
    images. For more information on the structure of the parameter file, see
    :ref:`Customizing the extraction <radiomics-customization-label>`.

    If supplied file does not match the requirements (i.e. unrecognized names or invalid values for a setting), a
    pykwalify error is raised.
    """
    dataDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'schemas'))
    schemaFile = os.path.join(dataDir, 'paramSchema.yaml')
    schemaFuncs = os.path.join(dataDir, 'schemaFuncs.py')
    c = pykwalify.core.Core(source_file=paramsFile, schema_files=[schemaFile], extensions=[schemaFuncs])
    params = c.validate()

    inputImages = params.get('inputImage', {})
    enabledFeatures = params.get('featureClass', {})
    settings = params.get('setting', {})
    voxelSettings = params.get('voxelWise', {})

    self.logger.debug("Parameter file parsed. Applying settings")

    if len(inputImages) == 0:
      self.inputImages = {'Original': {}}
    else:
      self.inputImages = inputImages

    self.logger.debug("Enabled input images: %s", self.inputImages)

    if len(enabledFeatures) == 0:
      self.enabledFeatures = {}
    else:
      self.enabledFeatures = enabledFeatures

    self.logger.debug("Enabled features: %s", enabledFeatures)

    # Set default settings and update with and changed settings contained in kwargs
    self.settings = self._getDefaultSettings()
    self.settings.update(settings)
    self.settings.update(voxelSettings)  # Update with voxel-wise specific settings

    self.logger.debug("Settings: %s", settings)

  def enableAllInputImages(self):
    """
    Enable all possible input images without any custom settings.
    """
    self.logger.debug('Enabling all input image types')
    for imageType in getInputImageTypes():
      self.inputImages[imageType] = {}
    self.logger.debug('Enabled input images types: %s', self.inputImages)

  def disableAllInputImages(self):
    """
    Disable all input images.
    """
    self.logger.debug('Disabling all input image types')
    self.inputImages = {}

  def enableInputImageByName(self, inputImage, enabled=True, customArgs=None):
    r"""
    Enable or disable specified input image. If enabling input image, optional custom settings can be specified in
    customArgs.

    Current possible input images are:

    - Original: No filter applied
    - Wavelet: Wavelet filtering, yields 8 decompositions per level (all possible combinations of applying either
      a High or a Low pass filter in each of the three dimensions.
      See also :py:func:`~radiomics.imageoperations.getWaveletImage`
    - LoG: Laplacian of Gaussian filter, edge enhancement filter. Emphasizes areas of gray level change, where sigma
      defines how coarse the emphasised texture should be. A low sigma emphasis on fine textures (change over a
      short distance), where a high sigma value emphasises coarse textures (gray level change over a large distance).
      See also :py:func:`~radiomics.imageoperations.getLoGImage`
    - Square: Takes the square of the image intensities and linearly scales them back to the original range.
      Negative values in the original image will be made negative again after application of filter.
    - SquareRoot: Takes the square root of the absolute image intensities and scales them back to original range.
      Negative values in the original image will be made negative again after application of filter.
    - Logarithm: Takes the logarithm of the absolute intensity + 1. Values are scaled to original range and
      negative original values are made negative again after application of filter.
    - Exponential: Takes the the exponential, where filtered intensity is e^(absolute intensity). Values are
      scaled to original range and negative original values are made negative again after application of filter.

    For the mathmetical formulas of square, squareroot, logarithm and exponential, see their respective functions in
    :ref:`imageoperations<radiomics-imageoperations-label>`
    (:py:func:`~radiomics.imageoperations.getSquareImage`,
    :py:func:`~radiomics.imageoperations.getSquareRootImage`,
    :py:func:`~radiomics.imageoperations.getLogarithmImage` and
    :py:func:`~radiomics.imageoperations.getExponentialImage`,
    respectively).
    """
    if inputImage not in getInputImageTypes():
      self.logger.warning('Input image type %s is not recognized', inputImage)
      return

    if enabled:
      if customArgs is None:
        customArgs = {}
        self.logger.debug('Enabling input image type %s (no additional custom settings)', inputImage)
      else:
        self.logger.debug('Enabling input image type %s (additional custom settings: %s)', inputImage, customArgs)
      self.inputImages[inputImage] = customArgs
    elif inputImage in self.inputImages:
      self.logger.debug('Disabling input image type %s', inputImage)
      del self.inputImages[inputImage]
    self.logger.debug('Enabled input images types: %s', self.inputImages)

  def enableInputImages(self, **inputImages):
    """
    Enable input images, with optionally custom settings, which are applied to the respective input image.
    Settings specified here override those in kwargs. This only applies to settings that operate on the filter level or
    feature class level.

    Updates current settings: If necessary, enables input image. Always overrides custom settings specified
    for input images passed in inputImages.
    To disable input images, use :py:func:`enableInputImageByName` or :py:func:`disableAllInputImages`
    instead.

    :param inputImages: dictionary, key is imagetype (original, wavelet or log) and value is custom settings
      (dictionary)
    """
    self.logger.debug('Updating enabled input images types with %s', inputImages)
    self.inputImages.update(inputImages)
    self.logger.debug('Enabled input images types: %s', self.inputImages)

  def enableAllFeatures(self):
    """
    Enable all classes and all features.
    """
    self.logger.debug('Enabling all features in all feature classes')
    for featureClassName in self.featureClasses.keys():
      self.enabledFeatures[featureClassName] = []
    self.logger.debug('Enabled features: %s', self.enabledFeatures)

  def disableAllFeatures(self):
    """
    Disable all classes.
    """
    self.logger.debug('Disabling all feature classes')
    self.enabledFeatures = {}

  def enableFeatureClassByName(self, featureClass, enabled=True):
    """
    Enable or disable all features in given class.
    """
    if featureClass not in self.featureClasses.keys():
      self.logger.warning('Feature class %s is not recognized', featureClass)
      return

    if enabled:
      self.logger.debug('Enabling all features in class %s', featureClass)
      self.enabledFeatures[featureClass] = []
    elif featureClass in self.enabledFeatures:
      self.logger.debug('Disabling feature class %s', featureClass)
      del self.enabledFeatures[featureClass]
    self.logger.debug('Enabled features: %s', self.enabledFeatures)

  def enableFeaturesByName(self, **enabledFeatures):
    """
    Specify which features to enable. Key is feature class name, value is a list of enabled feature names.

    To enable all features for a class, provide the class name with an empty list or None as value.
    Settings for feature classes specified in enabledFeatures.keys are updated, settings for feature classes
    not yet present in enabledFeatures.keys are added.
    To disable the entire class, use :py:func:`disableAllFeatures` or :py:func:`enableFeatureClassByName` instead.
    """
    self.logger.debug('Updating enabled features with %s', enabledFeatures)
    self.enabledFeatures.update(enabledFeatures)
    self.logger.debug('Enabled features: %s', self.enabledFeatures)

  def execute(self, imageFilepath, maskFilepath, label=None):
    """
    Compute radiomics signature for provide image and mask combination. It comprises of the following steps:

    1. Image and mask are loaded and normalized/resampled if necessary.
    2. Validity of ROI is checked using :py:func:`~imageoperations.checkMask`, which also computes and returns the
       bounding box.
    3. Other enabled featureclasses are calculated using all specified input image types in ``inputImages``. Images are
       cropped to tumor mask (no padding) after application of any filter and before being passed to the feature class.
    4. The calculated features is returned as ``collections.OrderedDict``, with feature name as key and a Simple ITK
       image object as the value.

    :param imageFilepath: SimpleITK Image, or string pointing to image file location
    :param maskFilepath: SimpleITK Image, or string pointing to labelmap file location
    :param label: Integer, value of the label for which to extract features. If not specified, last specified label
        is used. Default label is 1.
    :returns: dictionary containing calculated signature ("<filter>_<featureClass>_<featureName>":value).
    """
    # Enable or disable C extensions for high performance matrix calculation. Only logs a message (INFO) when setting is
    # successfully changed. If an error occurs, full-python mode is forced and a warning is logged.
    radiomics.enableCExtensions(self.settings['enableCExtensions'])
    if self.geometryTolerance != self.settings.get('geometryTolerance'):
      self._setTolerance()

    if label is not None:
      self.settings['label'] = label

    self.logger.info('Calculating features with label: %d', self.settings['label'])
    self.logger.debug('Enabled input images types: %s', self.inputImages)
    self.logger.debug('Enabled features: %s', self.enabledFeatures)
    self.logger.debug('Current settings: %s', self.settings)

    # 1. Load the image and mask
    featureVector = collections.OrderedDict()
    image, mask = self.loadImage(imageFilepath, maskFilepath)

    if image is None or mask is None:
      # No features can be extracted, return the empty featureVector
      return featureVector

    # 2. Check whether loaded mask contains a valid ROI for feature extraction and get bounding box
    boundingBox, correctedMask = imageoperations.checkMask(image, mask, **self.settings)

    # Update the mask if it had to be resampled
    if correctedMask is not None:
      mask = correctedMask

    if boundingBox is None:
      # Mask checks failed, do not extract features and return the empty featureVector
      return featureVector

    self.logger.debug('Image and Mask loaded and valid, starting extraction')

    # If maskedKernel is false, the crop function should include extra padding to handle the edge voxels correctly
    if not self.settings.get('maskedKernel', False):
      padDistance = self.settings.get('kernelDistance', 1)
    else:
      padDistance = 0

    # 4. Calculate enabled feature classes using enabled input image types
    # Make generators for all enabled input image types
    self.logger.debug('Creating input image type iterator')
    imageGenerators = []
    inputImageTypes = getInputImageTypes()
    for imageType, customKwargs in six.iteritems(self.inputImages):
      if imageType in inputImageTypes:
        args = self.settings.copy()
        args.update(customKwargs)
        self.logger.info('Adding image type "%s" with settings: %s' % (imageType, str(args)))
        imageGenerators = chain(imageGenerators, inputImageTypes[imageType](image, **args))

    self.logger.debug('Extracting features')
    # Calculate features for all (filtered) images in the generator
    for inputImage, inputImageName, inputKwargs in imageGenerators:
      self.logger.info('Calculating features for %s image', inputImageName)
      inputImage, inputMask = imageoperations.cropToTumorMask(inputImage, mask, boundingBox, padDistance)
      featureVector.update(self.computeFeatures(inputImage, inputMask, inputImageName, **inputKwargs))

    self.logger.debug('Features extracted')

    return featureVector

  def loadImage(self, ImageFilePath, MaskFilePath):
    """
    Preprocess the image and labelmap.
    If ImageFilePath is a string, it is loaded as SimpleITK Image and assigned to image,
    if it already is a SimpleITK Image, it is just assigned to image.
    All other cases are ignored (nothing calculated).
    Equal approach is used for assignment of mask using MaskFilePath.

    If normalizing is enabled image is first normalized before any resampling is applied.

    If resampling is enabled, both image and mask are resampled and cropped to the tumor mask (with additional
    padding as specified in padDistance) after assignment of image and mask.
    """
    self.logger.info('Loading image and mask')
    if isinstance(ImageFilePath, six.string_types) and os.path.exists(ImageFilePath):
      image = sitk.ReadImage(ImageFilePath)
    elif isinstance(ImageFilePath, sitk.SimpleITK.Image):
      image = ImageFilePath
    else:
      self.logger.warning('Error reading image Filepath or SimpleITK object')
      return None, None  # this function is expected to always return a tuple of 2 elements

    if isinstance(MaskFilePath, six.string_types) and os.path.exists(MaskFilePath):
      mask = sitk.ReadImage(MaskFilePath)
    elif isinstance(MaskFilePath, sitk.SimpleITK.Image):
      mask = MaskFilePath
    else:
      self.logger.warning('Error reading mask Filepath or SimpleITK object')
      return None, None  # this function is expected to always return a tuple of 2 elements

    # This point is only reached if image and mask loaded correctly
    if self.settings['normalize']:
      image = imageoperations.normalizeImage(image, self.settings['normalizeScale'], self.settings['removeOutliers'])

    if self.settings['interpolator'] is not None and self.settings['resampledPixelSpacing'] is not None:
      image, mask = imageoperations.resampleImage(image, mask,
                                                  self.settings['resampledPixelSpacing'],
                                                  self.settings['interpolator'],
                                                  self.settings['label'],
                                                  self.settings['padDistance'])

    return image, mask

  def computeFeatures(self, image, mask, inputImageName, **kwargs):
    """
    Compute signature using image, mask, \*\*kwargs settings.

    This function computes the signature for just the passed image (original or derived), it does not preprocess or
    apply a filter to the passed image. Features / Classes to use for calculation of signature are defined in
    self.enabledFeatures. See also :py:func:`enableFeaturesByName`.

    .. note::

      shape descriptors are not available for voxel-wise extraction and therefore not calculated. If this class is
      present in the enabled features, a warning is logged and shape features are skipped.
    """
    featureVector = collections.OrderedDict()

    # Calculate feature classes
    for featureClassName, enabledFeatures in six.iteritems(self.enabledFeatures):
      # Handle calculation of shape features not supported in voxel-wise extraction
      if featureClassName == 'shape':
        self.logger.warning('Shape features are not available for voxel-wise extraction')
        continue

      if featureClassName in self.featureClasses.keys():
        self.logger.info('Computing %s', featureClassName)

        featureClass = self.featureClasses[featureClassName](image, mask, **kwargs)

        if enabledFeatures is None or len(enabledFeatures) == 0:
          featureClass.enableAllFeatures()
        else:
          for feature in enabledFeatures:
            featureClass.enableFeatureByName(feature)

        featureClass.calculateVoxels()
        for (featureName, featureValue) in six.iteritems(featureClass.featureValues):
          newFeatureName = '%s_%s_%s' % (inputImageName, featureClassName, featureName)
          featureVector[newFeatureName] = featureValue

    return featureVector
