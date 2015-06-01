public Mat onCameraFrame(CvCameraViewFrame inputFrame){
		//to do 
		curGray = inputFrame.gray();
		curRGB = inputFrame.rgba();
		
		Mat T = new Mat(2, 3, CvType.CV_64F);
		Mat last_T = new Mat(2, 3, CvType.CV_64F);
		
		Mat cur2 = curRGB;
		//Mat last_T = new Mat();
		
		if(preGray != null && preRGB != null){
			
			int vertBorder = HORIZONTAL_BORDER_CROP * preGray.rows() / preGray.cols() ;
			
			MatOfPoint pre_corner = new MatOfPoint();
			MatOfPoint2f pre_corner2f = new MatOfPoint2f(), cur_corner2f = new MatOfPoint2f();
			MatOfPoint2f pre_corner2 = new MatOfPoint2f(), cur_corner2 = new MatOfPoint2f();
			MatOfByte status = new MatOfByte();
			MatOfFloat err = new MatOfFloat();
			
			Imgproc.goodFeaturesToTrack(preGray, pre_corner, 200, 0.01, 30.0);
			pre_corner.convertTo(pre_corner2f, CvType.CV_32FC2);
			Video.calcOpticalFlowPyrLK(preGray, curGray, pre_corner2f, cur_corner2f, status, err);
			for(int i = 0; i < status.toArray().length; i++){
				if(status.toArray()[i] != 0){
					pre_corner2.toArray()[i] = pre_corner2f.toArray()[i];
					cur_corner2.toArray()[i] = pre_corner2f.toArray()[i];
				}
			}
			
			T = Video.estimateRigidTransform(pre_corner2, cur_corner2, false);
			if(T.total() > 0){
				last_T.copyTo(T);
			}
			T.copyTo(last_T);
			
			double dx = T.get(0, 2)[0];
			double dy = T.get(1, 2)[0];
			double da = Math.atan2(T.get(1, 0)[0], T.get(0, 0)[0]);
			//mpre2CurTransformation.add(new TransformParam(dx, dy, da));
			
			mx += dx;
			my += dy;
			ma += da;
			
			mZ = new Trajectory(mx, my, ma);
			
			if(k == 1){
				mX = new Trajectory(0, 0, 0);
				mP = new Trajectory(1, 1, 1);
			}else{
				mX_ = mX;
				mP_ = mP;
				
				mK = Trajectory.div(mP_, Trajectory.plus(mP_, mR));
				mX = Trajectory.plus(mX_, Trajectory.multiply(mK, Trajectory.minus(mZ, mX_)));
				Trajectory temp = new Trajectory(1, 1, 1);
				mP = Trajectory.multiply(Trajectory.minus(temp, mK), mP_);
			}
			
			double diff_x = mX.x - mx;
			double diff_y = mX.y - my;
			double diff_a = mX.a - ma;
			
			dx = dx + diff_x;
			dy = dy + diff_y;
			da = da + diff_a;
			
			double[] setTemp = new double[1];
			setTemp[0] = Math.cos(da);
			T.put(0, 0, setTemp);
			setTemp[0] = - Math.sin(da);
			T.put(0, 1, setTemp);
			setTemp[0] = Math.sin(da);
			T.put(1, 0, setTemp);
			setTemp[0] = Math.cos(da);
			T.put(1, 1, setTemp);
			
			setTemp[0] = dx;
			T.put(0, 2, setTemp);
			setTemp[0] = dy;
			T.put(1, 2, setTemp);
			
			Imgproc.warpAffine(preRGB, cur2, T, preRGB.size());
			cur2 = cur2.submat(new Range(vertBorder, cur2.rows() - vertBorder), new Range(HORIZONTAL_BORDER_CROP, cur2.cols() - HORIZONTAL_BORDER_CROP));
			
			Imgproc.resize(cur2, cur2, curRGB.size());
			
			curRGB.copyTo(preRGB);
			curGray.copyTo(preGray);
			
			//smooth out the trajectory using an average window
			//Vector<Trajectory> smoothedTrajectory = new Vector();
			
		}
		
		preGray = curGray;
		preRGB = curRGB;
		return cur2;
	}