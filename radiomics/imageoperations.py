import SimpleITK as sitk
import numpy, pywt, logging
from itertools import chain

def getHistogram(binwidth, parameterValues):
  # Start binning form the first value lesser than or equal to the minimum value and evenly dividable by binwidth
  lowBound = min(parameterValues) - (min(parameterValues) % binwidth)
  # Add + binwidth to ensure the maximum value is included in the range generated by numpu.arange
  highBound = max(parameterValues) + binwidth

  binedges = numpy.arange(lowBound, highBound, binwidth)

  binedges[-1] += 1 # ensures that max(self.targertVoxelArray) is binned to upper bin by numpy.digitize

  return numpy.histogram(parameterValues, bins=binedges)

def binImage(binwidth, parameterValues, parameterMatrix = None, parameterMatrixCoordinates = None):
  histogram = getHistogram(binwidth, parameterValues)
  parameterMatrix[parameterMatrixCoordinates] = numpy.digitize(parameterValues, histogram[1])

  return parameterMatrix, histogram

def generateAngles(size, maxDistance= 1):
  """
  Generate all possible angles from distance 1 until maxDistance in 3D.
  E.g. for d = 1, 13 angles are generated (representing the 26-connected region).
  For d = 2, 13 + 49 = 62 angles are generated (representing the 26 connected region for distance 1, and the 98
  connected region for distance 2)

  Impossible angles (where 'neighbouring' voxels will always be outside delineation) are deleted.

  :param size: dimensions (z, x, y) of the bounding box of the tumor mask.
  :param maxDistance: [1] Maximum distance between center voxel and neighbour
  :return: numpy array with shape (N, 3), where N is the number of unique angles
  """

  angles = []

  for z in xrange(1, maxDistance+1):
    angles.append((0, 0, z))
    for y in xrange(-maxDistance, maxDistance+1):
      angles.append((0, z, y))
      for x in xrange(-maxDistance, maxDistance+1):
        angles.append((z, y, x))

  angles = numpy.array(angles)

  angles = numpy.delete(angles, numpy.where(numpy.min(size - numpy.abs(angles), 1) <= 0), 0)

  return angles

def cropToTumorMask(imageNode, maskNode):
  """
  Create a sitkImage of the segmented region of the image based on the input label.

  Create a sitkImage of the labelled region of the image, cropped to have a
  cuboid shape equal to the ijk boundaries of the label.

  Returns both the cropped version of the image and the cropped version of the labelmap.
  """

  oldMaskID = maskNode.GetPixelID()
  maskNode = sitk.Cast(maskNode, sitk.sitkInt32)
  size = numpy.array(maskNode.GetSize())

  #Determine bounds
  lsif = sitk.LabelStatisticsImageFilter()
  lsif.Execute(imageNode, maskNode)
  bb = lsif.GetBoundingBox(1)

  ijkMinBounds = bb[0::2]
  ijkMaxBounds = size - bb[1::2] - 1

  #Crop Image
  cif = sitk.CropImageFilter()
  cif.SetLowerBoundaryCropSize(ijkMinBounds)
  cif.SetUpperBoundaryCropSize(ijkMaxBounds)
  croppedImageNode = cif.Execute(imageNode)
  croppedMaskNode = cif.Execute(maskNode)

  croppedMaskNode = sitk.Cast(croppedMaskNode, oldMaskID)

  return croppedImageNode, croppedMaskNode

def resampleImage(imageNode, maskNode, resampledPixelSpacing, interpolator=sitk.sitkBSpline, padDistance= 5):
  """Resamples image or label to the specified pixel spacing (The default interpolator is Bspline)

  'imageNode' is a SimpleITK Object, and 'resampledPixelSpacing' is the output pixel spacing.
  Enumerator references for interpolator:
  0 - sitkNearestNeighbor
  1 - sitkLinear
  2 - sitkBSpline
  3 - sitkGaussian
  """

  if imageNode == None or maskNode == None:
    return None

  oldSpacing = numpy.array(imageNode.GetSpacing())

  # If current spacing is equal to resampledPixelSpacing, no interpolation is needed,
  # crop/pad image using cropTumorMaskToCube
  if numpy.array_equal(oldSpacing, resampledPixelSpacing):
    return cropToTumorMask(imageNode, maskNode)

  # Determine bounds of cropped volume in terms of original Index coordinate space
  lssif = sitk.LabelShapeStatisticsImageFilter()
  lssif.Execute(maskNode)
  bb = numpy.array(lssif.GetBoundingBox(1))  # LBound and size of the bounding box, as (L_X, L_Y, L_Z, S_X, S_Y, S_Z)

  # Do not resample in those directions where labelmap spans only one slice.
  oldSize = bb[3:]
  resampledPixelSpacing = numpy.where(oldSize != 1, resampledPixelSpacing, oldSpacing)

  spacingRatio = oldSpacing / resampledPixelSpacing

  # Determine bounds of cropped volume in terms of new Index coordinate space,
  # round down for lowerbound and up for upperbound to ensure entire segmentation is captured (prevent data loss)
  # Pad with an extra .5 to prevent data loss in case of upsampling. For Ubound this is (-1 + 0.5 = -0.5)
  bbNewLBound = numpy.floor( (bb[:3] - 0.5) * spacingRatio - padDistance)
  bbNewUBound = numpy.ceil( (bb[:3] + bb[3:] - 0.5) * spacingRatio + padDistance)

  # Ensure resampling is not performed outside bounds of original image
  maxUbound = numpy.ceil(numpy.array(imageNode.GetSize()) * spacingRatio) - 1
  bbNewLBound = numpy.where(bbNewLBound < 0, 0, bbNewLBound)
  bbNewUBound = numpy.where(bbNewUBound > maxUbound, maxUbound, bbNewUBound)

  # Calculate the new size. Cast to int to prevent error in sitk.
  newSize = numpy.array(bbNewUBound - bbNewLBound + 1, dtype='int')

  # Determine continuous index of bbNewLBound in terms of the original Index coordinate space
  bbOriginalLBound = bbNewLBound / spacingRatio

  # Origin is located in center of first voxel, e.g. 1/2 of the spacing
  # from Corner, which corresponds to 0 in the original Index coordinate space.
  # The new spacing will be in 0 the new Index coordinate space. Here we use continuous
  # index to calculate where the new 0 of the new Index coordinate space (of the original volume
  # in terms of the original spacing, and add the minimum bounds of the cropped area to
  # get the new Index coordinate space of the cropped volume in terms of the original Index coordinate space.
  # Then use the ITK functionality to bring the contiuous index into the physical space (mm)
  newOriginIndex = numpy.array(.5*(resampledPixelSpacing-oldSpacing)/oldSpacing)
  newCroppedOriginIndex = newOriginIndex + bbOriginalLBound
  newOrigin = imageNode.TransformContinuousIndexToPhysicalPoint(newCroppedOriginIndex)

  oldImagePixelType = imageNode.GetPixelID()
  oldMaskPixelType = maskNode.GetPixelID()

  imageDirection = numpy.array(imageNode.GetDirection())

  rif = sitk.ResampleImageFilter()

  rif.SetOutputSpacing(resampledPixelSpacing)
  rif.SetOutputDirection(imageDirection)
  rif.SetSize(newSize)
  rif.SetOutputOrigin(newOrigin)

  rif.SetOutputPixelType(oldImagePixelType)
  rif.SetInterpolator(interpolator)
  resampledImageNode = rif.Execute(imageNode)

  rif.SetOutputPixelType(oldMaskPixelType)
  rif.SetInterpolator(sitk.sitkNearestNeighbor)
  resampledMaskNode = rif.Execute(maskNode)

  return resampledImageNode,resampledMaskNode

#
# Use the SimpleITK LaplacianRecursiveGaussianImageFilter
# on the input image with the given sigmaValue and return
# the filtered image.
# If sigmaValue is not greater than zero, return the input image.
#
def applyLoG(inputImage, sigmaValue=0.5):
  if sigmaValue > 0.0:
    size = numpy.array(inputImage.GetSize())
    spacing = numpy.array(inputImage.GetSpacing())
    if numpy.all(size >= numpy.ceil(sigmaValue / spacing) + 1):
      lrgif = sitk.LaplacianRecursiveGaussianImageFilter()
      lrgif.SetNormalizeAcrossScale(True)
      lrgif.SetSigma(sigmaValue)
      return lrgif.Execute(inputImage)
    else:
      logging.info('applyLoG: sigma/spacing + 1 must be greater than the size of the inputImage: %g', sigmaValue)
      return None
  else:
    logging.info('applyLoG: sigma must be greater than 0.0: %g', sigmaValue)
    return None

def applyThreshold(inputImage, lowerThreshold, upperThreshold, insideValue=None, outsideValue=0):
  # this mode is useful to generate the mask of thresholded voxels
  if insideValue:
    tif = sitk.BinaryThresholdImageFilter()
    tif.SetInsideValue(insideValue)
    tif.SetLowerThreshold(lowerThreshold)
    tif.SetUpperThreshold(upperThreshold)
  else:
    tif = sitk.ThresholdImageFilter()
    tif.SetLower(lowerThreshold)
    tif.SetUpper(upperThreshold)
  tif.SetOutsideValue(outsideValue)
  return tif.Execute(inputImage)

def swt3(inputImage, wavelet="coif1", level=1, start_level=0):
  matrix = sitk.GetArrayFromImage(inputImage)
  matrix = numpy.asarray(matrix)
  data = matrix.copy()
  if data.ndim != 3:
    raise ValueError("Expected 3D data array")

  original_shape = matrix.shape
  adjusted_shape = tuple([dim+1 if dim % 2 != 0 else dim for dim in original_shape])
  data = numpy.resize(data, adjusted_shape)

  if not isinstance(wavelet, pywt.Wavelet):
    wavelet = pywt.Wavelet(wavelet)

  for i in range(0, start_level):
    H, L = decompose_i(data, wavelet)
    LH, LL = decompose_j(L, wavelet)
    LLH, LLL = decompose_k(LL, wavelet)

    data = LLL.copy()

  ret = []
  for i in range(start_level, start_level + level):
    H, L = decompose_i(data, wavelet)

    HH, HL = decompose_j(H, wavelet)
    LH, LL = decompose_j(L, wavelet)

    HHH, HHL = decompose_k(HH, wavelet)
    HLH, HLL = decompose_k(HL, wavelet)
    LHH, LHL = decompose_k(LH, wavelet)
    LLH, LLL = decompose_k(LL, wavelet)

    data = LLL.copy()

    dec = {'HHH': HHH,
           'HHL': HHL,
           'HLH': HLH,
           'HLL': HLL,
           'LHH': LHH,
           'LHL': LHL,
           'LLH': LLH}
    for decName, decImage in dec.iteritems():
      decTemp = decImage.copy()
      decTemp= numpy.resize(decTemp, original_shape)
      sitkImage = sitk.GetImageFromArray(decTemp)
      sitkImage.CopyInformation(inputImage)
      dec[decName] = sitkImage

    ret.append(dec)

  data= numpy.resize(data, original_shape)
  approximation = sitk.GetImageFromArray(data)
  approximation.CopyInformation(inputImage)

  return approximation, ret

def decompose_i(data, wavelet):
  #process in i:
  H, L = [], []
  i_arrays = chain.from_iterable(numpy.transpose(data,(0,1,2)))
  for i_array in i_arrays:
    cA, cD = pywt.swt(i_array, wavelet, level=1, start_level=0)[0]
    H.append(cD)
    L.append(cA)
  H = numpy.hstack(H).reshape(data.shape)
  L = numpy.hstack(L).reshape(data.shape)
  return H, L

def decompose_j(data, wavelet):
  #process in j:
  H, L = [], []
  j_arrays = chain.from_iterable(numpy.transpose(data,(0,1,2)))
  for j_array in j_arrays:
    cA, cD = pywt.swt(j_array, wavelet, level=1, start_level=0)[0]
    H.append(cD)
    L.append(cA)
  H = numpy.asarray( [slice.T for slice in numpy.split(numpy.vstack(H), data.shape[0])] )
  L = numpy.asarray( [slice.T for slice in numpy.split(numpy.vstack(L), data.shape[0])] )
  return H, L

def decompose_k(data, wavelet):
  #process in k:
  H, L = [], []
  k_arrays = chain.from_iterable(numpy.transpose(data,(1,2,0)))
  for k_array in k_arrays:
    cA, cD = pywt.swt(k_array, wavelet, level=1, start_level=0)[0]
    H.append(cD)
    L.append(cA)
  H = numpy.dstack(H).reshape(data.shape)
  L = numpy.dstack(L).reshape(data.shape)
  return H, L
