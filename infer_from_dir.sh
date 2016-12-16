DIR=demos
while read p; do
    INFILE=${DIR}/${p}
    OUTFILE=${DIR}/${p}_out.jpg
    if [ ! -f ${OUTFILE} ]; then
#         echo ${OUTFILE}
        th computeProposals.lua models -img ${INFILE} -limg ${OUTFILE}
    fi
done <${DIR}/jpgs.txt
