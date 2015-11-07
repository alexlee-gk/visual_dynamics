for f in *.h5; do
	echo "data/$f" > "${f%.*}.txt"
done