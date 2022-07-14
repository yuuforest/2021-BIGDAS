package Problem.D2;

import java.util.Scanner;

public class Prob1959 {
    public static void main(String args[]) throws Exception{
		
		Scanner sc = new Scanner(System.in);
		
		int T = sc.nextInt();

		for (int i = 1; i < T+1; i++) {

			int A_len = sc.nextInt();
			int B_len = sc.nextInt();

			int[] A = new int[A_len];
			int[] B = new int[B_len];

			for (int j = 0; j < A_len; j++) A[j] = sc.nextInt();
			for (int j = 0; j < B_len; j++) B[j] = sc.nextInt();

			int Max = 0;

			System.out.println(A[0] + ", " + A[1] + ", " + A[2]);

			if(A_len == B_len) for (int k = 0; k < A_len; k++) Max += A[k]*B[k];
			else if(A_len < B_len) {
				for (int k = 0; k < (B_len - A_len + 1); k++) {
					int sum = 0;
					for (int l = 0; l < A_len; l++) sum += A[l]*B[k+l];
					Max = Math.max(sum, Max);
				}
			} else {
				for (int k = 0; k < (A_len - B_len + 1); k++) {
					int sum = 0;
					for (int l = 0; l < B_len; l++) sum += B[l]*A[k+l];
					Max = Math.max(sum, Max);
				}
			}
			System.out.println("#" + i + " " + Max);
		}
		sc.close();
	}
}
