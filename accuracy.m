function acc = accuracy(Label1,Label2)
%Label1:��ʵ��ǩ Label2:ӳ���ı�ǩ

T= Label1==Label2;
acc=sum(T)/length(Label2);

end