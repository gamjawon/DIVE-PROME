import 'package:freezed_annotation/freezed_annotation.dart';

part 'search_result.freezed.dart';

@freezed
abstract class SearchResult<T> with _$SearchResult<T> {
  const factory SearchResult({required List<T> items}) = _SearchResult<T>;
}
