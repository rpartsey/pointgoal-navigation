DEST_DIR=/private/home/maksymets/pointgoal-navigation/data/vo_datasets_3m/gibson/train
mkdir -p $DEST_DIR

ONEM_DIR=/private/home/maksymets/pointgoal-navigation/data/vo_datasets2/gibson/train
for filename in $ONEM_DIR/*.json
do
#  echo "$DEST_DIR/1m_$(basename $filename)"
  cp "$filename" "$DEST_DIR/1m_$(basename $filename)"
done

TWOM_DIR=/private/home/maksymets/pointgoal-navigation/data/vo_datasets_2m/gibson/train
for filename in $TWOM_DIR/*.json
do
  # echo "$DEST_DIR/2m_$(basename $filename)"
  cp "$filename" "$DEST_DIR/2m_$(basename $filename)"
done;
