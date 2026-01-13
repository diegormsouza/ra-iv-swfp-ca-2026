#-----------------------------------------------------------------------------------------------------------
# Training: Python and GOES-R Imagery: Function with general functions
#-----------------------------------------------------------------------------------------------------------
# Required modules
from netCDF4 import Dataset              # Read / Write NetCDF4 files
import os                                # Miscellaneous operating system interfaces
import numpy as np                       # Import the Numpy package
import colorsys                          # To make convertion of colormaps
import boto3                             # Amazon Web Services (AWS) SDK for Python
from botocore import UNSIGNED            # boto3 config
from botocore.config import Config       # boto3 config
import math                              # Mathematical functions
import time as t                         # Time access and conversions
from datetime import datetime, timedelta # Basic Dates and time types
from osgeo import osr                    # Python bindings for GDAL
from osgeo import gdal                   # Python bindings for GDAL
import warnings
import requests
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")
gdal.PushErrorHandler('CPLQuietErrorHandler')

#-----------------------------------------------------------------------------------------------------------

def download_GOES(timestamp, satellite, product, band, path_dest, scan_params=None, force_download=False, check_adjacent_hours=False):
    """
    Download files from NOAA GOES-16, GOES-17, GOES-18, or GOES-19 S3 bucket for a given date, time, satellite, product, and band.
    
    Args:
        timestamp (str): Date and time in format 'YYYY-MM-DD HH:MM' (e.g., '2025-10-08 12:00') for ABI/SUVI/SEIS/MAG; 
                         'YYYY-MM-DD HH:MM:SS' (e.g., '2025-10-08 12:00:20') for GLM.
        satellite (str): Satellite identifier ('G16', 'G17', 'G18', or 'G19' for GOES-16, 17, 18, or 19).
        product (str): Product type (e.g., 'ABI-L2-CMIPM', 'ABI-L1b-RadM', 'GLM-L2-LCFA', 'SUVI-L1b-Fe093', 'SEIS-L1b-EHIS', 'MAG-L1b-GEOF').
        band (int or str): ABI band number (e.g., 13) for band-specific products, or None for non-band products (e.g., GLM-L2-LCFA, SUVI-L1b-Fe093, SEIS-L1b-EHIS, MAG-L1b-GEOF).
        path_dest (str): Local directory to save the file.
        scan_params (dict, optional): Override scan settings, e.g., {'mode': '3', 'sector': 'M1'} for Mode 3, Meso 1 (ignored for GLM/SUVI/SEIS/MAG).
        force_download (bool, optional): If True, re-download files even if they exist locally. Default: False.
        check_adjacent_hours (bool, optional): If True and no SUVI/SEIS/MAG files are found for the exact minute, check adjacent hours (±1 hour). Default: False.
    
    Returns:
        str or list or None: Path of the downloaded file (str) if one file is downloaded (e.g., './samples/file.nc' for ABI/GLM/MAG),
                            list of paths (e.g., ['path/to/file.nc', 'path/to/file.fits']) if multiple files are downloaded (e.g., SUVI),
                            or None if no files match the exact minute (SUVI/SEIS/MAG/ABI), closest within ±2 minutes (CONUS), or seconds (GLM).
    """
    def fetch_s3_objects(bucket_name, prefix, s3_client):
        all_objects = []
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    all_objects.extend(page['Contents'])
        except Exception as e:
            print(f"Error listing S3 objects for {bucket_name}/{prefix}: {e}")
        return all_objects

    valid_satellites = ['G16', 'G17', 'G18', 'G19']
    if satellite not in valid_satellites:
        print(f"Invalid satellite: {satellite}. Must be one of {valid_satellites}.")
        return None

    satellite_num = satellite[1:]
    os.makedirs(path_dest, exist_ok=True)

    # Parse input timestamp
    has_seconds = len(timestamp.replace('-', '').replace(' ', '').replace(':', '')) == 14  # YYYYMMDDHHMMSS
    try:
        if has_seconds:
            dt_input = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            timestamp_prefix = dt_input.strftime('%Y%j%H%M%S')
        else:
            dt_input = datetime.strptime(timestamp + ':00', '%Y-%m-%d %H:%M:%S')
            timestamp_prefix = dt_input.strftime('%Y%j%H%M')
    except ValueError as e:
        print(f"Invalid date format: {timestamp}. Expected 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD HH:MM:SS' for GLM.")
        return None

    year = dt_input.strftime('%Y')
    day_of_year = dt_input.strftime('%j')
    hour = dt_input.strftime('%H')

    bucket_name = f'noaa-goes{satellite_num}'
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    mode = scan_params.get('mode', '6') if scan_params else '6'
    sector = scan_params.get('sector', None) if scan_params else None

    # Check product type
    is_mesoscale = product.endswith('M')
    is_conus = product.endswith('C') and not is_mesoscale
    is_glm_product = product == 'GLM-L2-LCFA'
    is_suvi_product = product.startswith('SUVI-L1b-')
    is_seis_product = product.startswith('SEIS-L1b-')
    is_mag_product = product == 'MAG-L1b-GEOF'

    # Validate timestamp format for GLM
    if is_glm_product and not has_seconds:
        print(f"Warning: GLM product {product} prefers 'YYYY-MM-DD HH:MM:SS' format; using 20-second window for {timestamp}.")

    # Define prefixes to check
    prefixes = [f'{product}/{year}/{day_of_year}/{hour}/']
    if (is_suvi_product or is_seis_product or is_mag_product) and check_adjacent_hours:
        # Add previous and next hour
        dt_prev = dt_input - timedelta(hours=1)
        dt_next = dt_input + timedelta(hours=1)
        prefixes.extend([
            f'{product}/{dt_prev.strftime("%Y")}/{dt_prev.strftime("%j")}/{dt_prev.strftime("%H")}/',
            f'{product}/{dt_next.strftime("%Y")}/{dt_next.strftime("%j")}/{dt_next.strftime("%H")}/'
        ])

    all_objects = []
    for prefix in prefixes:
        objects = fetch_s3_objects(bucket_name, prefix, s3_client)
        all_objects.extend(objects)

    if not all_objects:
        print(f'No files found for product: {product}, date: {timestamp}, GOES-{satellite_num}')
        return None

    # Check if product is band-specific (ABI products)
    is_band_specific = False
    for obj in all_objects:
        file_name = obj['Key'].split('/')[-1]
        for m in range(1, 7):
            for b in range(1, 17):
                if f'-M{m}C{b:02d}' in file_name:
                    is_band_specific = True
                    break
            if is_band_specific:
                break
        if is_band_specific:
            break

    if is_band_specific:
        if band is None:
            print(f"Warning: Product {product} requires a band number (e.g., 13). Trying first available band.")
            band = '01'
        band_str = f'C{int(band):02d}'
    else:
        if band is not None:
            print(f"Warning: Product {product} does not use bands; ignoring band={band}.")
        band_str = ''

    # Check product type
    if is_glm_product:
        if sector is not None:
            print(f"Warning: GLM product {product} does not use sectors; ignoring sector={sector}.")
        if band is not None:
            print(f"Warning: GLM product {product} does not use bands; ignoring band={band}.")
        sectors = ['']  # GLM has no sectors
        band_str = ''
    elif is_suvi_product or is_seis_product or is_mag_product:
        if sector is not None:
            print(f"Warning: {'SUVI' if is_suvi_product else 'SEIS' if is_seis_product else 'MAG'} product {product} does not use sectors; ignoring sector={sector}.")
        if band is not None:
            print(f"Warning: {'SUVI' if is_suvi_product else 'SEIS' if is_seis_product else 'MAG'} product {product} does not use bands; ignoring band={band}.")
        sectors = ['']  # SUVI/SEIS/MAG have no sectors
        band_str = ''
    elif is_mesoscale:
        sectors = [sector] if sector in ['M1', 'M2'] else ['M1', 'M2']
    else:
        sectors = ['']  # Non-mesoscale ABI products (e.g., CMIPF, CMIPC)

    possible_modes = [mode, '3', '4', '6'] if not (is_glm_product or is_suvi_product or is_seis_product or is_mag_product) else ['']  # No modes for GLM/SUVI/SEIS/MAG
    matching_files = []  # List to store both .nc and .fits for SUVI, or multiple .nc for SEIS/MAG/ABI
    used_mode = mode if not (is_glm_product or is_suvi_product or is_seis_product or is_mag_product) else ''
    used_sector = ''
    attempted_patterns = []
    closest_file = None
    closest_time_diff = timedelta(minutes=10)  # Initialize with max allowed for non-CONUS

    for try_mode in possible_modes:
        for sector in sectors:
            if is_mesoscale:
                product_with_sector = f'{product[:-1]}{sector}'  # e.g., 'ABI-L2-CMIPM' -> 'ABI-L2-CMIPM1'
            else:
                product_with_sector = product
            target_pattern = f'OR_{product_with_sector}{"-M" + try_mode if try_mode else ""}{band_str}_G{satellite_num}_s'
            attempted_patterns.append(target_pattern)

            for obj in all_objects:
                key = obj['Key']
                file_name_full = key.split('/')[-1]
                if not file_name_full.startswith(target_pattern):
                    continue

                # Extract timestamp from filename
                timestamp_start = key.find('_s') + 2
                timestamp_end = key.find('_e', timestamp_start)
                if timestamp_end == -1:
                    continue
                file_timestamp_full = key[timestamp_start : timestamp_end]
                if len(file_timestamp_full) < 14:
                    continue
                file_timestamp_str = file_timestamp_full[:-1]  # Remove decimal

                # Matching logic based on product type
                if is_glm_product:
                    # GLM: Match exact timestamp (including seconds) or closest within 20 seconds if no seconds provided
                    file_timestamp_prefix = file_timestamp_str[:13] if has_seconds else file_timestamp_str[:11]
                    if has_seconds:
                        if file_timestamp_prefix == timestamp_prefix:
                            matching_files = [key]  # Only one file (.nc) for GLM
                            used_sector = sector
                            break
                    else:
                        try:
                            dt_file = datetime.strptime(file_timestamp_str, '%Y%j%H%M%S')
                            time_diff = abs(dt_file - dt_input)
                            if time_diff <= timedelta(seconds=20):  # Closest within 20 seconds
                                if not matching_files or time_diff < timedelta(seconds=20):
                                    matching_files = [key]
                                    used_sector = sector
                        except ValueError:
                            continue
                elif is_conus:
                    # CONUS: Find closest file within ±2 minutes
                    try:
                        dt_file = datetime.strptime(file_timestamp_str, '%Y%j%H%M%S')
                        time_diff = abs(dt_file - dt_input)
                        if time_diff <= timedelta(minutes=2):  # Within ±2 minutes
                            if closest_file is None or time_diff < closest_time_diff:
                                closest_file = key
                                closest_time_diff = time_diff
                                used_mode = try_mode
                                used_sector = sector
                    except ValueError:
                        continue
                elif is_suvi_product or is_seis_product or is_mag_product or product.startswith('ABI-'):
                    # SUVI, SEIS, MAG, and other ABI: Match exact minute
                    file_timestamp_prefix = file_timestamp_str[:11]  # e.g., '20252811200'
                    if file_timestamp_prefix == timestamp_prefix[:11]:
                        matching_files.append(key)  # Collect .nc and .fits for SUVI, .nc for SEIS/MAG/ABI
                    else:
                        continue
                else:
                    # Other products: Find closest within 10 minutes
                    try:
                        dt_file = datetime.strptime(file_timestamp_str, '%Y%j%H%M%S')
                        time_diff = abs(dt_file - dt_input)
                        if time_diff < timedelta(minutes=10):
                            if not matching_files or time_diff < closest_time_diff:
                                matching_files = [key]
                                closest_time_diff = time_diff
                                used_mode = try_mode
                                used_sector = sector
                    except ValueError:
                        continue

            if matching_files or closest_file:
                break
        if matching_files or closest_file:
            break

    # Combine matching files and closest file for CONUS
    if is_conus and closest_file:
        matching_files = [closest_file]
    elif not matching_files and not closest_file:
        print(f"No matching file found for product: {product}, band: {band}, date: {timestamp}, GOES-{satellite_num}")
        print(f"Debug: Attempted patterns: {attempted_patterns}")
        if is_mesoscale and sector:
            print(f"Note: Specified sector {sector} may not have band {band} files. Try omitting sector or using 'M2'.")
        return None

    if used_mode != mode and not (is_glm_product or is_suvi_product or is_seis_product or is_mag_product):
        print(f"Note: Used mode M{used_mode} instead of default M{mode} for better match.")

    # Download all matching files
    downloaded_files = []
    for matching_file in matching_files:
        file_name = matching_file.split('/')[-1]
        file_path = os.path.join(path_dest, file_name)

        # Check if file exists
        if os.path.exists(file_path) and not force_download:
            print(f'File {file_path} already exists')
            downloaded_files.append(file_path)
            continue

        # Download the file
        sector_str = f"{used_sector}, " if used_sector else ""
        mode_str = f"mode M{used_mode}" if used_mode else ""
        print(f'Downloading file {file_path} from GOES-{satellite_num}')
        try:
            s3_client.download_file(bucket_name, matching_file, file_path)
            print(f'Successfully downloaded {file_path}')  # Success message
            downloaded_files.append(file_path)
        except Exception as e:
            print(f"Error downloading {file_name} from GOES-{satellite_num}: {e}")

    # Return single file path if only one file, list if multiple, or None if empty
    if not downloaded_files:
        return None
    elif len(downloaded_files) == 1:
        return downloaded_files[0]
    else:
        return downloaded_files

########################################################################################################################
# download_mimic_tpw: Descarga archivo MIMIC TPW v2 NetCDF desde CIMSS
# URL base: https://bin.ssec.wisc.edu/pub/mtpw2/data/
# Formato: compYYYYMMDD.HHMM00.nc
# Autor: Adaptado de Diego Souza (INPE/CGCT/DISSM)
########################################################################################################################
def download_mimic_tpw(date_str, output_dir='./samples'):
    """
    Descarga el archivo MIMIC TPW v2 para una fecha/hora específica.
    
    Parámetros:
        date_str (str): Fecha y hora en formato 'YYYY-MM-DD HH:MM' o 'YYYY-MM-DD HH:MM:SS'
        output_dir (str): Directorio donde guardar el archivo (default: './samples')
    
    Retorna:
        str: Ruta completa del archivo descargado
    """
    import os
    import requests
    from datetime import datetime
    
    # Normalizamos la fecha/hora
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M')  # si incluye segundos, ajusta el formato
    year = dt.strftime('%Y')
    month = dt.strftime('%m')
    day = dt.strftime('%d')
    hour = dt.strftime('%H')
    minute = dt.strftime('%M')
    
    # Construimos el nombre del archivo
    file_name = f'comp{year}{month}{day}.{hour}{minute}00.nc'
    
    # URL completa
    url = f'https://bin.ssec.wisc.edu/pub/mtpw2/data/{year}{month}/{file_name}'
    
    # Directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, file_name)
    
    print(f"Descargando MIMIC TPW v2 desde CIMSS: {file_name}")
    print(f"URL: {url}")
    
    # Descarga
    try:
        r = requests.get(url, allow_redirects=True, timeout=30)
        r.raise_for_status()  # Lanza error si falla
        
        with open(local_path, 'wb') as f:
            f.write(r.content)
        
        print(f"Descarga exitosa: {local_path}")
        return local_path
    
    except Exception as e:
        print(f"Error al descargar {file_name}: {str(e)}")
        return None

def download_alpw(datetime_str: str, output_dir="./samples", verbose=True):
    """
    Download the BHP-ALPW file whose filename contains the END TIME eYYYYMMDDHH
    using the official S3 XML listing (ListObjectsV2).
    """

    # ── Parse datetime ───────────────────────────────────────────────────────
    try:
        target = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        if target.minute != 0:
            raise ValueError("The time must end in :00")
    except Exception as e:
        raise ValueError(f"Expected format: YYYY-MM-DD HH:00 → {e}")

    year  = target.strftime("%Y")
    month = target.strftime("%m")
    day   = target.strftime("%d")
    hour  = target.strftime("%H")

    end_token = f"e{year}{month}{day}{hour}"

    prefix = f"JPSS_Blended_Products/BHP_ALPW/{year}/{month}/{day}/"

    list_url = (
        "https://noaa-jpss.s3.amazonaws.com/"
        f"?list-type=2&prefix={prefix}"
    )

    if verbose:
        print("Searching for files in:")
        print(prefix)

    # ── List S3 objects ──────────────────────────────────────────────────────
    r = requests.get(list_url, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.content)

    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    filename = None
    for key in root.findall(".//s3:Key", namespace):
        name = key.text
        if name.endswith(".nc") and end_token in name:
            filename = os.path.basename(name)
            break

    if filename is None:
        raise FileNotFoundError(
            f"No file found with END TIME {end_token}\n"
            f"Prefix: {prefix}"
        )

    file_url = "https://noaa-jpss.s3.amazonaws.com/" + prefix + filename

    if verbose:
        print("\nFile found:")
        print(filename)
        print(f"URL: {file_url}")

    # ── Download ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with requests.get(file_url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

    if verbose:
        print("\n✓ Download completed")
        print(f"File saved at: {output_path}")
        print(f"File size: {os.path.getsize(output_path):,} bytes")

    return output_path
	
def loadCPT(path):

    try:
        f = open(path)
    except:
        print ("File ", path, "not found")
        return None

    lines = f.readlines()

    f.close()

    x = np.array([])
    r = np.array([])
    g = np.array([])
    b = np.array([])

    colorModel = 'RGB'

    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x=np.append(x,float(ls[0]))
            r=np.append(r,float(ls[1]))
            g=np.append(g,float(ls[2]))
            b=np.append(b,float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

        x=np.append(x,xtemp)
        r=np.append(r,rtemp)
        g=np.append(g,gtemp)
        b=np.append(b,btemp)

    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
        r[i] = rr ; g[i] = gg ; b[i] = bb

    if colorModel == 'RGB':
        r = r/255.0
        g = g/255.0
        b = b/255.0

    xNorm = (x - x[0])/(x[-1] - x[0])

    red   = []
    blue  = []
    green = []

    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])

    colorDict = {'red': red, 'green': green, 'blue': blue}

    return colorDict
#-----------------------------------------------------------------------------------------------------------
def download_CMI(yyyymmddhhmn, band, path_dest):

  os.makedirs(path_dest, exist_ok=True)

  year = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%Y')
  day_of_year = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%j')
  hour = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%H')
  min = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%M')

  # AMAZON repository information 
  # https://noaa-goes16.s3.amazonaws.com/index.html
  bucket_name = 'noaa-goes16'
  product_name = 'ABI-L2-CMIPF'

  # Initializes the S3 client
  s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  #-----------------------------------------------------------------------------------------------------------
  # File structure
  prefix = f'{product_name}/{year}/{day_of_year}/{hour}/OR_{product_name}-M6C{int(band):02.0f}_G16_s{year}{day_of_year}{hour}{min}'

  # Seach for the file on the server
  s3_result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter = "/")

  #-----------------------------------------------------------------------------------------------------------
  # Check if there are files available
  if 'Contents' not in s3_result: 
    # There are no files
    print(f'No files found for the date: {yyyymmddhhmn}, Band-{band}')
    return -1
  else:
    # There are files
    for obj in s3_result['Contents']: 
      key = obj['Key']
      # Print the file name
      file_name = key.split('/')[-1].split('.')[0]

      # Download the file
      if os.path.exists(f'{path_dest}/{file_name}.nc'):
        print(f'File {path_dest}/{file_name}.nc exists')
      else:
        print(f'Downloading file {path_dest}/{file_name}.nc')
        s3_client.download_file(bucket_name, key, f'{path_dest}/{file_name}.nc')
  return f'{file_name}'

#-----------------------------------------------------------------------------------------------------------
def download_PROD(yyyymmddhhmn, product_name, path_dest):

  os.makedirs(path_dest, exist_ok=True)

  year = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%Y')
  day_of_year = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%j')
  hour = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%H')
  min = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%M')

  # AMAZON repository information 
  # https://noaa-goes16.s3.amazonaws.com/index.html
  bucket_name = 'noaa-goes16'

  # Initializes the S3 client
  s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  #-----------------------------------------------------------------------------------------------------------
  # File structure
  prefix = f'{product_name}/{year}/{day_of_year}/{hour}/OR_{product_name}-M6_G16_s{year}{day_of_year}{hour}{min}'

  # Seach for the file on the server
  s3_result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter = "/")

  #-----------------------------------------------------------------------------------------------------------
  # Check if there are files available
  if 'Contents' not in s3_result: 
    # There are no files
    print(f'No files found for the date: {yyyymmddhhmn}, Product-{product_name}')
    return -1
  else:
    # There are files
    for obj in s3_result['Contents']: 
      key = obj['Key']
      # Print the file name
      file_name = key.split('/')[-1].split('.')[0]

      # Download the file
      if os.path.exists(f'{path_dest}/{file_name}.nc'):
        print(f'File {path_dest}/{file_name}.nc exists')
      else:
        print(f'Downloading file {path_dest}/{file_name}.nc')
        s3_client.download_file(bucket_name, key, f'{path_dest}/{file_name}.nc')
  return f'{file_name}'

#-----------------------------------------------------------------------------------------------------------
def download_GLM(yyyymmddhhmnss, path_dest):

  os.makedirs(path_dest, exist_ok=True)

  year = datetime.strptime(yyyymmddhhmnss, '%Y%m%d%H%M%S').strftime('%Y')
  day_of_year = datetime.strptime(yyyymmddhhmnss, '%Y%m%d%H%M%S').strftime('%j')
  hour = datetime.strptime(yyyymmddhhmnss, '%Y%m%d%H%M%S').strftime('%H')
  min = datetime.strptime(yyyymmddhhmnss, '%Y%m%d%H%M%S').strftime('%M')
  seg = datetime.strptime(yyyymmddhhmnss, '%Y%m%d%H%M%S').strftime('%S')

  # AMAZON repository information 
  # https://noaa-goes16.s3.amazonaws.com/index.html
  bucket_name = 'noaa-goes16'

  # Initializes the S3 client
  s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  #-----------------------------------------------------------------------------------------------------------
  # File structure
  product_name = "GLM-L2-LCFA"
  prefix = f'{product_name}/{year}/{day_of_year}/{hour}/OR_{product_name}_G16_s{year}{day_of_year}{hour}{min}{seg}'

  # Seach for the file on the server
  s3_result = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter = "/")

  #-----------------------------------------------------------------------------------------------------------
  # Check if there are files available
  if 'Contents' not in s3_result: 
    # There are no files
    print(f'No files found for the date: {yyyymmddhhmnss}, Product-{product_name}')
    return -1
  else:
    # There are files
    for obj in s3_result['Contents']: 
      key = obj['Key']
      # Print the file name
      file_name = key.split('/')[-1].split('.')[0]

      # Download the file
      if os.path.exists(f'{path_dest}/{file_name}.nc'):
        print(f'File {path_dest}/{file_name}.nc exists')
      else:
        print(f'Downloading file {path_dest}/{file_name}.nc')
        s3_client.download_file(bucket_name, key, f'{path_dest}/{file_name}.nc')
  return f'{file_name}'

#-----------------------------------------------------------------------------------------------------------
# Functions to convert lat / lon extent to array indices 
def geo2grid(lat, lon, nc):

    # Apply scale and offset 
    xscale, xoffset = nc.variables['x'].scale_factor, nc.variables['x'].add_offset
    yscale, yoffset = nc.variables['y'].scale_factor, nc.variables['y'].add_offset
    
    x, y = latlon2xy(lat, lon)
    col = (x - xoffset)/xscale
    lin = (y - yoffset)/yscale
    return int(lin), int(col)

def latlon2xy(lat, lon):
    # goes_imagery_projection:semi_major_axis
    req = 6378137 # meters
    #  goes_imagery_projection:inverse_flattening
    invf = 298.257222096
    # goes_imagery_projection:semi_minor_axis
    rpol = 6356752.31414 # meters
    e = 0.0818191910435
    # goes_imagery_projection:perspective_point_height + goes_imagery_projection:semi_major_axis
    H = 42164160 # meters
    # goes_imagery_projection: longitude_of_projection_origin
    lambda0 = -1.308996939

    # Convert to radians
    latRad = lat * (math.pi/180)
    lonRad = lon * (math.pi/180)

    # (1) geocentric latitude
    Phi_c = math.atan(((rpol * rpol)/(req * req)) * math.tan(latRad))
    # (2) geocentric distance to the point on the ellipsoid
    rc = rpol/(math.sqrt(1 - ((e * e) * (math.cos(Phi_c) * math.cos(Phi_c)))))
    # (3) sx
    sx = H - (rc * math.cos(Phi_c) * math.cos(lonRad - lambda0))
    # (4) sy
    sy = -rc * math.cos(Phi_c) * math.sin(lonRad - lambda0)
    # (5)
    sz = rc * math.sin(Phi_c)

    # x,y
    x = math.asin((-sy)/math.sqrt((sx*sx) + (sy*sy) + (sz*sz)))
    y = math.atan(sz/sx)

    return x, y

# Function to convert lat / lon extent to GOES-16 extents
def convertExtent2GOESProjection(extent):
    # GOES-16 viewing point (satellite position) height above the earth
    GOES16_HEIGHT = 35786023.0
    # GOES-16 longitude position
    GOES16_LONGITUDE = -75.0
	
    a, b = latlon2xy(extent[1], extent[0])
    c, d = latlon2xy(extent[3], extent[2])
    return (a * GOES16_HEIGHT, c * GOES16_HEIGHT, b * GOES16_HEIGHT, d * GOES16_HEIGHT)

#-----------------------------------------------------------------------------------------------------------
# Function to reproject the data
def reproject(file_name, ncfile, array, extent, undef):

    # Read the original file projection and configure the output projection
    source_prj = osr.SpatialReference()
    source_prj.ImportFromProj4(ncfile.GetProjectionRef())

    target_prj = osr.SpatialReference()
    target_prj.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
   
    # Reproject the data
    GeoT = ncfile.GetGeoTransform()
    driver = gdal.GetDriverByName('MEM')
    raw = driver.Create('raw', array.shape[0], array.shape[1], 1, gdal.GDT_Float32)
    raw.SetGeoTransform(GeoT)
    raw.GetRasterBand(1).WriteArray(array)

    # Define the parameters of the output file  
    kwargs = {'format': 'netCDF', \
            'srcSRS': source_prj, \
            'dstSRS': target_prj, \
            'outputBounds': (extent[0], extent[3], extent[2], extent[1]), \
            'outputBoundsSRS': target_prj, \
            'outputType': gdal.GDT_Float32, \
            'srcNodata': undef, \
            'dstNodata': 'nan', \
            'resampleAlg': gdal.GRA_NearestNeighbour}

    # Write the reprojected file on disk
    gdal.Warp(file_name, raw, **kwargs)

#-----------------------------------------------------------------------------------------------------------
def exportImage(image,path):
    driver = gdal.GetDriverByName('netCDF')
    return driver.CreateCopy(path,image,0)
#-----------------------------------------------------------------------------------------------------------
def getGeoT(extent, nlines, ncols):
    # Compute resolution based on data dimension
    resx = (extent[2] - extent[0]) / ncols
    resy = (extent[3] - extent[1]) / nlines
    return [extent[0], resx, 0, extent[3] , 0, -resy]
#-----------------------------------------------------------------------------------------------------------
def getScaleOffset(path, variable):
    nc = Dataset(path, mode='r')

    if (variable == "BCM") or (variable == "Phase") or (variable == "Smoke") or (variable == "Dust") or (variable == "Mask") or (variable == "Power"):
        scale  = 1
        offset = 0
    else:
        scale = nc.variables[variable].scale_factor
        offset = nc.variables[variable].add_offset
    nc.close()

    return scale, offset
#-----------------------------------------------------------------------------------------------------------
def remap(path, variable, extent, resolution):

    # Read the image
    file = Dataset(path)

    # Read the semi major axis
    a = file.variables['goes_imager_projection'].semi_major_axis

    # Read the semi minor axis
    b = file.variables['goes_imager_projection'].semi_minor_axis

    # Calculate the image extent
    h = file.variables['goes_imager_projection'].perspective_point_height
    x1 = file.variables['x_image_bounds'][0] * h
    x2 = file.variables['x_image_bounds'][1] * h
    y1 = file.variables['y_image_bounds'][1] * h
    y2 = file.variables['y_image_bounds'][0] * h

    # Read the central longitude
    longitude = file.variables['goes_imager_projection'].longitude_of_projection_origin

    # Default scale
    scale = 1

    # Default offset
    offset = 0

    # GOES Extent (satellite projection) [llx, lly, urx, ury]
    GOES_EXTENT = [x1, y1, x2, y2]

    # Setup NetCDF driver
    gdal.SetConfigOption('GDAL_NETCDF_BOTTOMUP', 'NO')

    if not (variable == "DQF"):
        # Read scale/offset from file
        scale, offset = getScaleOffset(path, variable)

    connectionInfo = 'HDF5:\"' + path + '\"://' + variable

    #print(connectionInfo)

    # Read the datasat
    raw = gdal.Open(connectionInfo)

    # Define KM_PER_DEGREE
    KM_PER_DEGREE = 111.32

    # GOES Spatial Reference System
    sourcePrj = osr.SpatialReference()
    sourcePrj.ImportFromProj4('+proj=geos +h=' + str(h) + ' ' + '+a=' + str(a) + ' ' + '+b=' + str(b) + ' ' + '+lon_0=' + str(longitude) + ' ' + '+sweep=x')

    # Lat/lon WSG84 Spatial Reference System
    targetPrj = osr.SpatialReference()
    targetPrj.ImportFromProj4('+proj=latlong +datum=WGS84')

    # Setup projection and geo-transformation
    raw.SetProjection(sourcePrj.ExportToWkt())
    raw.SetGeoTransform(getGeoT(GOES_EXTENT, raw.RasterYSize, raw.RasterXSize))

    # Compute grid dimension
    sizex = int(((extent[2] - extent[0]) * KM_PER_DEGREE) / resolution)
    sizey = int(((extent[3] - extent[1]) * KM_PER_DEGREE) / resolution)

    # Get memory driver
    memDriver = gdal.GetDriverByName('MEM')

    # Create grid
    grid = memDriver.Create('grid', sizex, sizey, 1, gdal.GDT_Float32)

    # Setup projection and geo-transformation
    grid.SetProjection(targetPrj.ExportToWkt())
    grid.SetGeoTransform(getGeoT(extent, grid.RasterYSize, grid.RasterXSize))

    # Perform the projection/resampling
    print ('Remapping...')#, path)

    start = t.time()

    gdal.ReprojectImage(raw, grid, sourcePrj.ExportToWkt(), targetPrj.ExportToWkt(), gdal.GRA_NearestNeighbour, options=['NUM_THREADS=ALL_CPUS'])

    print ('Remap finished! Time:', round((t.time() - start),2), 'seconds')

    # Read grid data
    array = grid.ReadAsArray()

    # Mask fill values (i.e. invalid values)
    np.ma.masked_where(array, array == -1, False)

    # Read as uint16
    array = array.astype(np.uint16)

    # Apply scale and offset
    array = array * scale + offset

    # Get the raster
    grid.GetRasterBand(1).WriteArray(array)

    # GENERATE A NEW NETCDF FILE =================================
    connectionInfo = 'NETCDF:\"' + path + '\"://' + variable
    # Read the datasat
    img = gdal.Open(connectionInfo)
    metadata = img.GetMetadata()
    undef = float(metadata.get(variable + '#_FillValue'))
    # Define the parameters of the output file
    options = gdal.WarpOptions(format = 'netCDF',
    srcSRS = sourcePrj,
    dstSRS = targetPrj,
    outputBounds = (extent[0], extent[3], extent[2], extent[1]),
    outputBoundsSRS = targetPrj,
    outputType = gdal.GDT_Float32,
    srcNodata = undef,
    dstNodata = 'nan',
    xRes = resolution/100,
    yRes = resolution/100,
    resampleAlg = gdal.GRA_NearestNeighbour)

    # Write the reprojected file on disk
    gdal.Warp(f'{path[:-3]}_ret.nc', raw, options=options)
    #==============================================================

    # Close file
    raw = None; img=None

    return grid
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------



