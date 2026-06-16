#!/bin/bash
# Downloads selected WV videos to data/raw/wvdoh/
# Skips files already downloaded
# Shows progress per file
# Usage: bash scripts/download_wv_sample.sh

mkdir -p data/raw/wvdoh

if [ ! -f "data/raw/wvdoh/260325_135228_263_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260325_135228_263_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260325_135228_263_FH.MP4"
else
  echo "SKIP: 260325_135228_263_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260402_092126_486_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260402_092126_486_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260402_092126_486_FH.MP4"
else
  echo "SKIP: 260402_092126_486_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260331_093446_244_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260331_093446_244_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260331_093446_244_FH.MP4"
else
  echo "SKIP: 260331_093446_244_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/251009_121000_032_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/251009_121000_032_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/251009_121000_032_FH.MP4"
else
  echo "SKIP: 251009_121000_032_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260414_073606_302_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260414_073606_302_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260414_073606_302_FH.MP4"
else
  echo "SKIP: 260414_073606_302_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260401_065959_005_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260401_065959_005_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260401_065959_005_FH.MP4"
else
  echo "SKIP: 260401_065959_005_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260304_132844_571_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260304_132844_571_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260304_132844_571_FH.MP4"
else
  echo "SKIP: 260304_132844_571_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260305_143707_623_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260305_143707_623_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260305_143707_623_FH.MP4"
else
  echo "SKIP: 260305_143707_623_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260421_142457_104_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260421_142457_104_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260421_142457_104_FH.MP4"
else
  echo "SKIP: 260421_142457_104_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260327_072640_417_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260327_072640_417_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260327_072640_417_FH.MP4"
else
  echo "SKIP: 260327_072640_417_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260414_122656_280_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260414_122656_280_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260414_122656_280_FH.MP4"
else
  echo "SKIP: 260414_122656_280_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/251020_105333_049_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/251020_105333_049_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/251020_105333_049_FH.MP4"
else
  echo "SKIP: 251020_105333_049_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/251009_085256_013_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/251009_085256_013_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/251009_085256_013_FH.MP4"
else
  echo "SKIP: 251009_085256_013_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260320_135748_425_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260320_135748_425_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260320_135748_425_FH.MP4"
else
  echo "SKIP: 260320_135748_425_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260326_133800_563_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260326_133800_563_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260326_133800_563_FH.MP4"
else
  echo "SKIP: 260326_133800_563_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260325_131025_249_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260325_131025_249_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260325_131025_249_FH.MP4"
else
  echo "SKIP: 260325_131025_249_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260430_050311_610_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260430_050311_610_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260430_050311_610_FH.MP4"
else
  echo "SKIP: 260430_050311_610_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/250909_171012_933_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/250909_171012_933_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/250909_171012_933_FH.MP4"
else
  echo "SKIP: 250909_171012_933_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260416_102846_548_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260416_102846_548_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260416_102846_548_FH.MP4"
else
  echo "SKIP: 260416_102846_548_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260406_093305_004_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260406_093305_004_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260406_093305_004_FH.MP4"
else
  echo "SKIP: 260406_093305_004_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260326_134258_351_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260326_134258_351_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260326_134258_351_FH.MP4"
else
  echo "SKIP: 260326_134258_351_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260421_161843_282_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260421_161843_282_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260421_161843_282_FH.MP4"
else
  echo "SKIP: 260421_161843_282_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260402_120628_596_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260402_120628_596_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260402_120628_596_FH.MP4"
else
  echo "SKIP: 260402_120628_596_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/251016_115619_083_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/251016_115619_083_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/251016_115619_083_FH.MP4"
else
  echo "SKIP: 251016_115619_083_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260319_114115_065_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260319_114115_065_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260319_114115_065_FH.MP4"
else
  echo "SKIP: 260319_114115_065_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260505_105259_221_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260505_105259_221_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260505_105259_221_FH.MP4"
else
  echo "SKIP: 260505_105259_221_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/250908_080200_912_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/250908_080200_912_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/250908_080200_912_FH.MP4"
else
  echo "SKIP: 250908_080200_912_FH.MP4 already exists"
fi

if [ ! -f "data/raw/wvdoh/260327_123944_096_FH.MP4" ]; then
  wget -c --show-progress -O "data/raw/wvdoh/260327_123944_096_FH.MP4" "https://objectstorage.us-ashburn-1.oraclecloud.com/p/xI2c8lL1sjv13h6U9ebP5HwMmCtABlbJkTGWCLqvQ-v-S1oqJTr2b5aPHNth1jyD/n/idnz0hftfltw/b/wvu_dashcam_videos/o/260327_123944_096_FH.MP4"
else
  echo "SKIP: 260327_123944_096_FH.MP4 already exists"
fi
