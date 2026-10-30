colmap feature_extractor \
  --database_path progetto/database.db \
  --image_path progetto/images

colmap exhaustive_matcher \
  --database_path progetto/database.db

mkdir -p progetto/sparse
colmap mapper \
  --database_path progetto/database.db \
  --image_path progetto/images \
  --output_path progetto/sparse
