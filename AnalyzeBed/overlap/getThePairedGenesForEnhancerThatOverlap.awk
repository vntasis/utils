#!/usr/bin/awk -f

BEGIN{ OFS="\t" }
fname != FILENAME { fname=FILENAME; idx++ }
idx==1{
	enh=$1","$2","$3;
	if (!seen1[enh]++){ pairs1[enh]=$4 }
	else{
		pairs1[enh]=pairs1[enh]","$4
	}
}
idx==2{
	enh=$1","$2","$3;
	if (!seen2[enh]++){ pairs2[enh]=$4 }
	else{
		pairs2[enh]=pairs2[enh]","$4
	}
}
idx==3{
	enh1=$1","$2","$3;
	enh2=$4","$5","$6;
	n1=split(pairs1[enh1],g1,",");
	n2=split(pairs2[enh2],g2,",");
	for (i=1;i<=n1;i++){ genes[g1[i]]=1 };
	for (i=1;i<=n2;i++){ Ncommon+=genes[g2[i]] };
	for (i=1;i<=n2;i++){ genes[g2[i]]=1 };
	for (g in genes){ Nunion+=genes[g] };
	print pairs1[enh1], pairs2[enh2], Nunion, Ncommon, Ncommon/Nunion;
	Ncommon=0; Nunion=0;
	delete genes;
}

