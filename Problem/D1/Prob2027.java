package Problem.D1;

// 대각선 출력하기
public class Prob2027 {
    public static void main(String args[]) throws Exception
	{
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 97ms 19356KB

		sb.append("#++++\n");
		sb.append("+#+++\n");
		sb.append("++#++\n");
		sb.append("+++#+\n");
		sb.append("++++#\n");

		System.out.println(sb);
	}
}
