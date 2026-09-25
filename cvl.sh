#!/usr/bin/env bash
clear
# Step-2. GMT set up
# ==== 調整 GMT 預設參數 ====
gmt set FORMAT_GEO_MAP=ddd:mm:ssF \
			     MAP_FRAME_PEN dimgray \
			     MAP_FRAME_WIDTH 0.1c \
			     MAP_TITLE_OFFSET 1c \
			     MAP_ANNOT_OFFSET 0.1c \
					 MAP_FRAME_AXES=WesN \
			     MAP_TICK_PEN_PRIMARY thinner,dimgray \
			     MAP_GRID_PEN_PRIMARY thinner,dimgray \
			     MAP_GRID_PEN_SECONDARY thinnest,dimgray \
			     FONT_TITLE 12p,Palatino-Roman,black \
			     FONT_ANNOT_PRIMARY 12p,13,dimgray \
			     FONT_LABEL 12p,13,dimgray \
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#	-----------------------------------------------------------------------------------------------------------

	title=Topo_WCARS
	echo $title

#	Grilla
	GRD=@earth_relief_01m
#	volcano=/Users/dgicape/Documents/Gephysics/tectonics/afrk_volcano1.txt
	cvl=CVL.gmt
	all_craton=African_Cratons.gmt

#	plate=/Users/ghomsi/Documents/Solid_earth/Franck_mohamed_inversion/GMT_plot/PB2002_boundaries.gmt
#	meridian=/Users/ghomsi/Documents/Solid_earth/Franck_mohamed_inversion/GMT_plot/paralelos.txt

#	hotspot=/Users/ghomsi/Documents/Solid_earth/Franck_mohamed_inversion/GMT_plot/hotspots.gmt
	line1=Atlantic_FZ.gmt
	line2=sz.gmt
	line3=rt.gmt
	jos=j.gmt
#	chad=Chad_bassin.gmt
	Benue=Benue.gmt
#	Termite=termite.gmt	
#	iullemmeden=iullemmeden.gmt

#	Region y proyeccion geografica
	REGION=2/18/0/12 #1/24/2/25 #100/180/-50/10
	PROJ=M15c

# 	Archivos temporales
	CUT=temp_$title.nc
	SHADOW=+d
	color=temp_$title.cpt

	gmt set GMT_VERBOSE w

#	Dibujar mapa
#	-----------------------------------------------------------------------------------------------------------
	gmt begin $title png E300

#	Setear la region y proyeccion
	gmt basemap -R$REGION -J$PROJ -BWsNe
#	Recortar Grilla
	gmt grdcut $GRD -G$CUT -R$REGION


	gmt makecpt -Coslo -T-6000/0 -N -H >  $color
	#gmt makecpt -Cragray -T-6000/0 -N -H >  $color
	#gmt makecpt -Cdem2  -T0/5000  -H >> $color
	gmt makecpt -Cragray  -T0/5000  -H >> $color


	gmt grdimage $CUT -I$SHADOW -C$color



gmt basemap -Tdg22/23+w0.45i+l+f3 -F+gwhite@50 --FONT=12p,Helvetica

gmt basemap -LjRB+c50+w200k+f+o12.5c/0.40c+u -F+gwhite@50 --FONT=7.5p,Palatino-Roman,black

	gmt set MAP_ANNOT_OFFSET_SECONDARY = 0.14i
	gmt set MAP_TICK_LENGTH_PRIMARY = -0.20i
	gmt set MAP_FRAME_PEN = 1p\
	FONT_ANNOT_PRIMARY=15p,29,40/40/40\
	FONT_LABEL=15p,29,40/40/40

	gmt colorbar -DJBC+w7.5/0.418c+o0c/0.22c+e+h -C$color -Ba1+l"Topography" -By+l"(km)" -I -W0.001


	gmt basemap -Bxa2f1 -Bya4f2 #-BWesN

		gmt coast -Df -N1/thin,black #-I1/thin,STEELBLUE1

		#gmt coast -Df -W1/faint -CSTEELBLUE1

		gmt plot $all_craton  -A  -W1.75,black -Gp500/21:F107/107/107B- -t45

		gmt plot $cvl   -A  -W2.0p,red -Gp500/21:F107/107/107B- -t45
		
		gmt plot $jos   -A  -W2.0p,red -Gp500/21:F107/107/107B- -t45
		
		#gmt plot $cvl  -R-1/1/-1/1 -JX10c -W2p,red -S > polygon.ps
		
		gmt plot $line3 -Sf3.5c/0.2i+l+s+o1 -W1.0p,blue,+

		gmt plot $line2 -W2.0p,black,+
		
		gmt plot $line1 -W2.0p,firebrick3,

	#	gmt plot $chad -W0.75p,black,-

		gmt plot $Benue -W0.50p,black -Gp500/21:F107/107/107B- -t45

#		gmt plot $Termite -W0.22p,black,-

#		gmt plot $iullemmeden -W0.75p,black,-


#		gmt plot $volcano -Skvolcano/0.25 -Wthinnest -Ggreen@35 #-l"Volcano"

		#________________________________________________________________________#3
 gmt text -F+a+jLT+f8.75p,15,black -Givory@50 -Wthinnest -C+tO << TEXT2END
		17.75		11.5		40		#Chad lineament
TEXT2END




 gmt text -F+a+jLT+f13p,15,white=0.25p,black << TEXT2END
		20		9.25		17		CASZ
		5.42		17		0		WAMZ
TEXT2END

 gmt text -F+a+jLT+f10p,15,white=0.25p,red << TEXT4END
		11	7		35       CVL
TEXT4END

 
TEXT2END

 gmt text -F+a+jLT+f12.2p,13,black -Gorange@50 -Wthinnest -C+tO << TEXT5END
		12		14.5		0		CB
		14		23.75		0		MB
		20.5		21.25		0		AkB
		11		18		0		TB
		6		23		0		HM
		17		21		0		TM
		6.7		11.75		0		NSB
		8 18.75	0	A\357r
		2.2	15	0	lullemmeden
		3	14.46	0	Basin
TEXT5END

gmt text -F+a+jLT+f12p,13,tomato2 -Givory@50 -Wthinnest -C+tO  << TEXT5END
11		20.5		-55		WARS
19		7.25		17		CARS
TEXT5END





gmt inset begin -DjTL+w1.95i+o-0.75i/-0.75i
        gmt coast -JG11.5/23.5N/? -Rg -B -N1 -Wfaint -Ggray -A5000 -SSTEELBLUE1
				gmt plot -JG11.5/23.5N/? -Rg $all_craton  -A  -W1.25,red
    gmt inset end
#	-----------------------------------------------------------------------------------------------------------

gmt end show

	rm -f temp_* gmt.* .nc *.cpt $top
